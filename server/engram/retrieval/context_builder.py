"""Build tiered memory context for agent prompt loading."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.retrieval.budgets import (
    RecallBudget,
    budget_profile_for_source,
    recall_budget_for_profile,
    surface_for_source,
)
from engram.retrieval.memory_operations import (
    MemoryOperationSample,
    record_manager_memory_operation,
)
from engram.retrieval.packets import assemble_memory_packets
from engram.storage.protocols import ActivationStore, GraphStore

logger = logging.getLogger(__name__)
LOADED_STORE_CONTEXT_PACKET_SCOPE = "loaded_store_context"
DURABLE_CONTEXT_PACKET_SCOPE = "durable_context"
SESSION_RECENT_PACKET_SCOPE = "session_recent"
# Process-level durable pack cache: session-start get_context should be ~ms after
# the first cold pack within a short TTL. Independent of packet-cache config.
_DURABLE_CONTEXT_PROCESS_CACHE_TTL_SECONDS = 45.0
_DURABLE_CONTEXT_HARD_BUDGET_SECONDS = 2.0
# key -> (expires_at_monotonic, packets, entity_ids)
_durable_context_process_cache: dict[
    tuple[str, str],
    tuple[float, list[dict[str, Any]], set[str]],
] = {}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*([:=])\s*([^\s;,]+)"
)
_SECRET_TOKEN_RE = re.compile(r"\b(?:sk|pk|xox[baprs]|gh[pousr])[-_][A-Za-z0-9_\-]{8,}\b")
# Session-start / topic-less get_context still needs a probe that can find
# high-signal Decisions/Preferences without requiring the agent to craft a query.
_DEFAULT_DURABLE_CONTEXT_QUERY = "durable decisions preferences goals commitments corrections"


def invalidate_durable_context_cache(group_id: str | None = None) -> int:
    """Drop process-level durable context packs (after remember / graph mutation)."""
    if group_id is None:
        count = len(_durable_context_process_cache)
        _durable_context_process_cache.clear()
        return count
    keys = [key for key in _durable_context_process_cache if key[0] == group_id]
    for key in keys:
        del _durable_context_process_cache[key]
    return len(keys)


def _durable_context_cache_key(group_id: str, query: str) -> tuple[str, str]:
    return (group_id, query.casefold().strip())


def _load_durable_context_process_cache(
    group_id: str,
    query: str,
) -> tuple[list[dict[str, Any]], set[str]] | None:
    key = _durable_context_cache_key(group_id, query)
    entry = _durable_context_process_cache.get(key)
    if entry is None:
        return None
    expires_at, packets, entity_ids = entry
    if time.monotonic() >= expires_at or not packets:
        _durable_context_process_cache.pop(key, None)
        return None
    return [dict(packet) for packet in packets], set(entity_ids)


def _store_durable_context_process_cache(
    group_id: str,
    query: str,
    packets: Sequence[Mapping[str, Any]],
    entity_ids: set[str],
    *,
    ttl_seconds: float = _DURABLE_CONTEXT_PROCESS_CACHE_TTL_SECONDS,
) -> None:
    if not packets or ttl_seconds <= 0:
        return
    key = _durable_context_cache_key(group_id, query)
    _durable_context_process_cache[key] = (
        time.monotonic() + ttl_seconds,
        [dict(packet) for packet in packets],
        set(entity_ids),
    )
    # Bound process memory: keep most recent ~64 keys.
    if len(_durable_context_process_cache) > 64:
        oldest = min(
            _durable_context_process_cache.items(),
            key=lambda item: item[1][0],
        )[0]
        _durable_context_process_cache.pop(oldest, None)


async def build_api_context_surface(
    manager: Any,
    *,
    group_id: str,
    max_tokens: int = 2000,
    topic_hint: str | None = None,
    project_path: str | None = None,
    format: str = "structured",
    operation_source: str = "api_context",
) -> dict:
    """Build the REST context response from the shared context manager facade."""
    result = await build_mcp_context_surface(
        manager,
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
        operation_source=operation_source,
    )
    payload = {
        "context": result["context"],
        "entityCount": result["entity_count"],
        "factCount": result["fact_count"],
        "tokenEstimate": result["token_estimate"],
        "format": result.get("format", "structured"),
    }
    if "cached_packets" in result:
        payload["cachedPackets"] = result["cached_packets"]
    if "packet_cache" in result:
        payload["packetCache"] = result["packet_cache"]
    if "status" in result:
        payload["status"] = result["status"]
    if "budget" in result:
        budget = result["budget"]
        payload["budget"] = {
            "profile": budget.get("profile"),
            "surface": budget.get("surface"),
            "mode": budget.get("mode"),
            "maxWallMs": budget.get("max_wall_ms"),
            "durationMs": budget.get("duration_ms"),
            "budgetMiss": budget.get("budget_miss"),
            "timeout": budget.get("timeout"),
            "degraded": budget.get("degraded"),
            "skipReason": budget.get("skip_reason"),
        }
    if "lifecycle" in result:
        lifecycle = result["lifecycle"]
        payload["lifecycle"] = {
            "stage": lifecycle.get("stage"),
            "degraded": lifecycle.get("degraded"),
            "timeout": lifecycle.get("timeout"),
            "skipReason": lifecycle.get("skip_reason"),
        }
    if "diagnostics" in result:
        diagnostics = result["diagnostics"]
        stage_timings = diagnostics.get("stage_timings_ms") or {}
        payload["diagnostics"] = {
            "stageTimingsMs": {
                _camel_stage_name(str(key)): value
                for key, value in stage_timings.items()
                if isinstance(value, int | float)
            }
        }
    return payload


async def build_mcp_context_surface(
    manager: Any,
    *,
    group_id: str,
    max_tokens: int = 2000,
    topic_hint: str | None = None,
    project_path: str | None = None,
    format: str = "structured",
    operation_source: str = "mcp_context",
) -> dict:
    """Build the MCP context response from the shared context manager facade."""
    cfg = _manager_activation_config(manager)
    budget = recall_budget_for_profile(
        cfg,
        budget_profile_for_source(operation_source),
        surface=surface_for_source(operation_source),
        mode=operation_source,
        max_output_tokens=max_tokens,
    )
    timeout_seconds = budget.stage_timeout_seconds(budget.max_wall_ms)
    started = time.perf_counter()
    cache_topic_hint = _derive_context_topic_hint(topic_hint, project_path)
    context_topic_hint = _bounded_context_topic_hint(topic_hint, project_path)
    if timeout_seconds <= 0:
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await _record_context_timeout(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            budget=budget,
            status="skipped",
            skip_reason="skipped_budget",
            duration_ms=duration_ms,
            timeout=False,
        )
        return _context_timeout_payload(
            format=format,
            budget=budget,
            duration_ms=duration_ms,
            status="skipped",
            skip_reason="skipped_budget",
            timeout=False,
        )

    # Product path: session-start continuity should not depend on project files.
    # Prefer the same durable Decision/Preference rescue that explicit recall uses.
    durable_started = time.perf_counter()
    durable_payload = await _durable_context_payload_from_manager(
        manager,
        group_id=group_id,
        topic_hint=topic_hint or cache_topic_hint,
        project_path=project_path,
        format=format,
        budget=budget,
        started=started,
    )
    if durable_payload is not None:
        durable_payload.setdefault("diagnostics", {}).setdefault("stage_timings_ms", {})[
            "durable_context"
        ] = _elapsed_ms(durable_started)
        durable_cache_hit = bool((durable_payload.get("packet_cache") or {}).get("hit"))
        await record_manager_memory_operation(
            manager,
            group_id,
            MemoryOperationSample(
                operation="context",
                source=operation_source,
                mode=operation_source,
                status="ok",
                duration_ms=_elapsed_ms(started),
                budget_ms=budget.budget_ms,
                budget_tokens=budget.budget_tokens,
                cache_hit=durable_cache_hit,
                packet_count=len(durable_payload.get("cached_packets") or []),
                result_count=int(durable_payload.get("entity_count") or 0),
            ),
        )
        return durable_payload

    cached_payload = _cached_context_payload_from_manager(
        manager,
        group_id=group_id,
        topic_hint=cache_topic_hint,
        project_path=project_path,
        format=format,
        budget=budget,
        status="ok",
        duration_ms=_elapsed_ms(started),
        skip_reason=None,
        timeout=False,
        allow_session_recent_only=not _can_enrich_session_recent_context(
            manager,
            topic_hint=cache_topic_hint,
            project_path=project_path,
        ),
    )
    if cached_payload is not None:
        await record_manager_memory_operation(
            manager,
            group_id,
            MemoryOperationSample(
                operation="context",
                source=operation_source,
                mode=operation_source,
                status="ok",
                duration_ms=_elapsed_ms(started),
                budget_ms=budget.budget_ms,
                budget_tokens=budget.budget_tokens,
                cache_hit=True,
                packet_count=len(cached_payload.get("cached_packets") or []),
            ),
        )
        return cached_payload

    fast_project_context = _fast_project_context_requested(
        operation_source=operation_source,
        topic_hint=topic_hint,
        project_path=project_path,
    )
    if (
        _topic_specific_context_requested(
            topic_hint=cache_topic_hint,
            project_path=project_path,
        )
        or fast_project_context
    ):
        project_file_task = _start_project_file_context_fallback_task(
            topic_hint=cache_topic_hint,
            project_path=project_path,
            max_packets=5,
            reason=_project_file_fallback_reason("cache_relevance_miss"),
        )
        early_project_file_payload = _project_file_context_rescue_payload_from_task(
            project_file_task,
            manager,
            group_id=group_id,
            topic_hint=cache_topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status="ok",
            duration_ms=_elapsed_ms(started),
            skip_reason="cache_relevance_miss",
            timeout=False,
            started=started,
        )
        if early_project_file_payload is not None:
            await record_manager_memory_operation(
                manager,
                group_id,
                MemoryOperationSample(
                    operation="context",
                    source=operation_source,
                    mode=operation_source,
                    status="ok",
                    duration_ms=_elapsed_ms(started),
                    budget_ms=budget.budget_ms,
                    budget_tokens=budget.budget_tokens,
                    cache_hit=True,
                    packet_count=len(early_project_file_payload.get("cached_packets") or []),
                ),
            )
            return early_project_file_payload
        loaded_store_payload = None
        loaded_store_preflight_duration_ms = None
        loaded_store_task: asyncio.Task[dict[str, Any] | None] | None = None
        if not fast_project_context:
            loaded_store_preflight_started = time.perf_counter()
            loaded_store_task = asyncio.create_task(
                _loaded_store_context_payload_from_manager(
                    manager,
                    group_id=group_id,
                    topic_hint=cache_topic_hint,
                    project_path=project_path,
                    format=format,
                    budget=budget,
                    started=started,
                )
            )
            loaded_store_payload = await _await_loaded_store_context_preflight_task(
                loaded_store_task,
                project_file_task=project_file_task,
                soft_wait_seconds=_context_fast_preflight_soft_wait_seconds(cfg),
            )
            loaded_store_preflight_duration_ms = _elapsed_ms(loaded_store_preflight_started)
            if loaded_store_task is not None and not loaded_store_task.done():
                loaded_store_task.add_done_callback(_consume_loaded_store_context_task)
        if loaded_store_payload is not None:
            await record_manager_memory_operation(
                manager,
                group_id,
                MemoryOperationSample(
                    operation="context",
                    source=operation_source,
                    mode=operation_source,
                    status="ok",
                    duration_ms=loaded_store_payload["budget"]["duration_ms"],
                    budget_ms=budget.budget_ms,
                    budget_tokens=budget.budget_tokens,
                    cache_hit=False,
                    packet_count=len(loaded_store_payload.get("cached_packets") or []),
                ),
            )
            return loaded_store_payload

        project_file_payload = await _project_file_context_payload_from_task_or_manager(
            project_file_task,
            manager,
            group_id=group_id,
            topic_hint=cache_topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status="ok",
            duration_ms=_elapsed_ms(started),
            skip_reason="cache_relevance_miss",
            timeout=False,
            started=started,
            pre_fallback_stage_timings={
                "loaded_store_context_preflight": loaded_store_preflight_duration_ms,
            }
            if loaded_store_preflight_duration_ms is not None
            else None,
        )
        if project_file_payload is not None:
            await record_manager_memory_operation(
                manager,
                group_id,
                MemoryOperationSample(
                    operation="context",
                    source=operation_source,
                    mode=operation_source,
                    status="ok",
                    duration_ms=_elapsed_ms(started),
                    budget_ms=budget.budget_ms,
                    budget_tokens=budget.budget_tokens,
                    cache_hit=bool((project_file_payload.get("packet_cache") or {}).get("hit")),
                    packet_count=len(project_file_payload.get("cached_packets") or []),
                ),
            )
            return project_file_payload

    try:
        return await asyncio.wait_for(
            manager.get_context(
                group_id=group_id,
                max_tokens=max_tokens,
                topic_hint=context_topic_hint,
                project_path=project_path,
                format=format,
                operation_source=operation_source,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        await _record_context_timeout(
            manager,
            group_id=group_id,
            operation_source=operation_source,
            budget=budget,
            status="degraded",
            skip_reason="context_timeout",
            duration_ms=duration_ms,
            timeout=True,
        )
        cached_payload = _cached_context_payload_from_manager(
            manager,
            group_id=group_id,
            topic_hint=cache_topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status="degraded",
            duration_ms=duration_ms,
            skip_reason="context_timeout",
            timeout=True,
        )
        if cached_payload is not None:
            return cached_payload
        project_file_payload = _project_file_context_payload_from_manager(
            manager,
            group_id=group_id,
            topic_hint=cache_topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status="degraded",
            duration_ms=duration_ms,
            skip_reason="context_timeout",
            timeout=True,
        )
        if project_file_payload is not None:
            return project_file_payload
        return _context_timeout_payload(
            format=format,
            budget=budget,
            duration_ms=duration_ms,
            status="degraded",
            skip_reason="context_timeout",
            timeout=True,
        )


async def build_mcp_context_tool_surface(
    manager: Any,
    *,
    group_id: str,
    max_tokens: int = 2000,
    topic_hint: str | None = None,
    project_path: str | None = None,
    format: str = "structured",
    recall_middleware: Callable[..., Awaitable[None]],
    operation_source: str = "mcp_context",
) -> dict:
    """Build the MCP context tool payload and run read-tool middleware."""
    result = await build_mcp_context_surface(
        manager,
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
        operation_source=operation_source,
    )
    await recall_middleware(topic_hint or project_path or "", result, tool_name="get_context")
    return result


class MemoryContextBuilder:
    """Assemble the active memory context exposed through REST and MCP."""

    _CONTEXT_RECALL_TIMEOUT_SECONDS = 8.0
    _CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS = 3.0
    _CONTEXT_IDENTITY_LIMIT = 12
    _CONTEXT_PROJECT_RECALL_LIMIT = 5
    _CONTEXT_PROJECT_NEIGHBOR_LIMIT = 12
    _CONTEXT_PROJECT_ARTIFACT_LIMIT = 5
    _CONTEXT_RECENCY_LIMIT = 20

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cfg: ActivationConfig,
        recall: Callable[..., Awaitable[list[dict]]],
        list_intentions: Callable[..., Awaitable[list]],
        resolve_entity_name: Callable[[str, str], Awaitable[str]],
        publish_access_event: Callable[[str, str, str, str, str], Awaitable[None]],
        briefing_cache: dict[tuple[str, str | None], tuple[float, str]] | None = None,
        get_cached_packets: Callable[..., Any] | None = None,
        cache_packets: Callable[..., Any] | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._recall = recall
        self._list_intentions = list_intentions
        self._resolve_entity_name = resolve_entity_name
        self._publish_access_event = publish_access_event
        self._briefing_cache = briefing_cache if briefing_cache is not None else {}
        self._get_cached_packets = get_cached_packets
        self._cache_packets = cache_packets

    @property
    def briefing_cache(self) -> dict[tuple[str, str | None], tuple[float, str]]:
        return self._briefing_cache

    async def entity_to_context_data(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        now: float,
        detail_level: str = "full",
    ) -> dict:
        """Build context data dict for a single entity with activation and facts."""
        result: dict = {
            "name": name,
            "type": entity_type,
            "detail_level": detail_level,
            "id": entity_id,
        }

        if detail_level == "mention":
            result["activation"] = 0.0
            result["summary"] = None
            result["facts"] = []
            result["attributes"] = None
            return result

        from engram.activation.engine import compute_activation

        state = await self._activation.get_activation(entity_id)
        act = 0.0
        if state:
            act = compute_activation(state.access_history, now, self._cfg)
        result["activation"] = act
        result["summary"] = summary

        max_facts = 5 if detail_level == "full" else 2
        facts: list[str] = []
        rels = await self._graph.get_relationships(
            entity_id,
            active_only=True,
            group_id=group_id,
        )
        for relationship in rels[:max_facts]:
            src = await self._resolve_entity_name(relationship.source_id, group_id)
            tgt = await self._resolve_entity_name(relationship.target_id, group_id)
            facts.append(f"{src} {relationship.predicate} {tgt}")
        result["facts"] = facts

        if detail_level == "full":
            entity = await self._graph.get_entity(entity_id, group_id)
            result["attributes"] = entity.attributes if entity else None
        else:
            result["attributes"] = None

        return result

    @staticmethod
    def render_tier(header: str, entities: list[dict], facts: list[str]) -> str:
        """Render a single context tier as markdown with variable resolution."""
        lines = [header, ""]
        for entity_data in entities:
            detail = entity_data.get("detail_level", "full")

            if detail == "mention":
                lines.append(f"- {entity_data['name']} ({entity_data['type']})")
                continue

            summary = _redact_packet_text(str(entity_data.get("summary") or ""))
            summary_part = f" — {summary}" if summary else ""
            if detail == "full":
                attrs = entity_data.get("attributes")
                if attrs:
                    attr_parts = [
                        _redact_packet_text(f"{k}: {v}") for k, v in list(attrs.items())[:5]
                    ]
                    summary_part += f" [{', '.join(attr_parts)}]"
            lines.append(
                f"- {entity_data['name']} ({entity_data['type']}, "
                f"act={entity_data['activation']:.2f}){summary_part}"
            )
            for fact in entity_data.get("facts", []):
                lines.append(f"  - {_redact_packet_text(str(fact))}")

        entity_facts = set()
        for entity_data in entities:
            entity_facts.update(entity_data.get("facts", []))
        extra_facts = [fact for fact in facts if fact not in entity_facts]
        if extra_facts:
            for fact in extra_facts:
                lines.append(f"  - {_redact_packet_text(str(fact))}")
        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def invalidate_briefing_cache(self, group_id: str) -> None:
        """Clear briefing cache entries for the given group."""
        keys_to_remove = [key for key in self._briefing_cache if key[0] == group_id]
        for key in keys_to_remove:
            del self._briefing_cache[key]

    def template_briefing(
        self,
        structured_context: str,
        group_id: str,
        topic_hint: str | None,
        growth_stats: Mapping[str, Any] | None = None,
    ) -> str:
        """Render a brief deterministic narrative from structured context."""
        growth_key = (
            (
                int(growth_stats.get("episodes") or 0),
                int(growth_stats.get("cues") or 0),
                int(growth_stats.get("promotions") or 0),
            )
            if growth_stats
            else ()
        )
        cache_key = (group_id, topic_hint, growth_key)
        now = time.time()
        if cache_key in self._briefing_cache:
            timestamp, text = self._briefing_cache[cache_key]
            if now - timestamp < self._cfg.briefing_cache_ttl_seconds:
                return text

        sentences: list[str] = []
        if growth_stats:
            episodes = int(growth_stats.get("episodes") or 0)
            cues = int(growth_stats.get("cues") or 0)
            promotions = int(growth_stats.get("promotions") or 0)
            if episodes or cues or promotions:
                growth_line = (
                    f"Memory growth: {episodes} episodes, {cues} cue traces"
                    f"{f', {promotions} promoted to graph' if promotions else ''}"
                    " — compounding with each session you use Engram."
                )
                sentences.append(growth_line)
        tier1_lines: list[str] = []
        tier2_lines: list[str] = []
        tier3_lines: list[str] = []
        cached_lines: list[str] = []
        current_tier: list[str] | None = None

        for line in structured_context.split("\n"):
            stripped = line.strip()
            if "Cached Memory Packets" in stripped and stripped.startswith("#"):
                current_tier = cached_lines
            elif "Identity" in stripped and stripped.startswith("#"):
                current_tier = tier1_lines
            elif "Project" in stripped and stripped.startswith("#"):
                current_tier = tier2_lines
            elif ("Recent" in stripped or "Activity" in stripped) and stripped.startswith("#"):
                current_tier = tier3_lines
            elif "Intention" in stripped and stripped.startswith("#"):
                current_tier = None
            elif current_tier is not None and stripped.startswith("- "):
                current_tier.append(stripped[2:].strip())

        if cached_lines:
            sentences.append("Cached memory: " + "; ".join(cached_lines[:3]) + ".")
        if tier1_lines:
            sentences.append("Known context: " + "; ".join(tier1_lines[:3]) + ".")
        if tier2_lines:
            prefix = f"Currently working on {topic_hint}: " if topic_hint else "Current focus: "
            sentences.append(prefix + "; ".join(tier2_lines[:3]) + ".")
        if tier3_lines:
            sentences.append("Recent activity: " + "; ".join(tier3_lines[:3]) + ".")

        briefing = " ".join(sentences) if sentences else structured_context
        self._briefing_cache[cache_key] = (now, briefing)
        return briefing

    async def _collect_growth_stats(self, group_id: str) -> dict[str, Any]:
        get_stats = getattr(self._graph, "get_stats", None)
        if not callable(get_stats):
            return {}
        try:
            stats = get_stats(group_id)
            if inspect.isawaitable(stats):
                stats = await stats
        except Exception:
            logger.debug("growth stats collection failed", exc_info=True)
            return {}
        if not isinstance(stats, dict):
            return {}
        cue_metrics = stats.get("cue_metrics") or {}
        return {
            "episodes": int(stats.get("episodes") or 0),
            "cues": int(cue_metrics.get("cue_count") or 0),
            "promotions": int(cue_metrics.get("projected_cue_count") or 0),
            "entities": int(stats.get("entities") or 0),
        }

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        """Build a tiered markdown context summary of the most activated memories."""
        from engram.activation.engine import compute_activation

        stage_timings_ms: dict[str, float] = {}
        now = time.time()
        if project_path:
            project_dir = Path(project_path).expanduser()
            if project_dir.name and str(project_dir) != str(Path.home()) and not topic_hint:
                topic_hint = project_dir.name
        packet_started = time.perf_counter()
        cached_packets = self._load_cached_context_packets(
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
        )
        stage_timings_ms["packet_cache"] = _elapsed_ms(packet_started)
        cached_packet_text = self.render_cached_packets(cached_packets)
        seen_ids: set[str] = set()
        identity_limit = self._budgeted_entity_limit(
            max_tokens,
            default=self._CONTEXT_IDENTITY_LIMIT,
            tokens_per_entity=160,
            minimum=2,
        )
        project_recall_limit = self._budgeted_entity_limit(
            max_tokens,
            default=self._CONTEXT_PROJECT_RECALL_LIMIT,
            tokens_per_entity=160,
            minimum=1,
        )
        project_recall_timeout = min(
            self._CONTEXT_RECALL_TIMEOUT_SECONDS,
            self._budgeted_timeout(max_tokens, default=4.0),
        )
        project_lookup_timeout = min(
            self._CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS,
            self._budgeted_timeout(max_tokens, default=2.0),
        )
        project_neighbor_timeout = min(
            self._CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS,
            self._budgeted_timeout(max_tokens, default=3.0),
        )
        project_neighbor_limit = self._budgeted_entity_limit(
            max_tokens,
            default=self._CONTEXT_PROJECT_NEIGHBOR_LIMIT,
            tokens_per_entity=100,
            minimum=2,
        )
        project_artifact_limit = self._budgeted_entity_limit(
            max_tokens,
            default=self._CONTEXT_PROJECT_ARTIFACT_LIMIT,
            tokens_per_entity=120,
            minimum=2,
        )
        recency_limit = self._budgeted_entity_limit(
            max_tokens,
            default=self._CONTEXT_RECENCY_LIMIT,
            tokens_per_entity=100,
            minimum=2,
        )
        entity_context_timeout = min(
            0.05,
            self._budgeted_timeout(max_tokens, default=0.05),
        )

        project_entity_id: str | None = None
        if project_path:
            project_dir = Path(project_path).expanduser()
            if project_dir.name and str(project_dir) != str(Path.home()):
                project_lookup_timed_out = False
                project_lookup_started = time.perf_counter()
                try:
                    existing_projects = await asyncio.wait_for(
                        self._graph.find_entities(
                            name=project_dir.name,
                            entity_type="Project",
                            group_id=group_id,
                            limit=1,
                        ),
                        timeout=project_lookup_timeout,
                    )
                except TimeoutError:
                    project_lookup_timed_out = True
                    logger.debug(
                        "Context project lookup timed out for path=%r; "
                        "continuing without project graph expansion",
                        project_path,
                    )
                    existing_projects = []
                stage_timings_ms["project_lookup"] = _elapsed_ms(project_lookup_started)
                if existing_projects:
                    project_entity_id = existing_projects[0].id
                elif not project_lookup_timed_out:
                    project_entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                    project_entity = Entity(
                        id=project_entity_id,
                        name=project_dir.name,
                        entity_type="Project",
                        summary=f"Software project at {project_path}",
                        attributes={"project_path": str(project_dir)},
                        group_id=group_id,
                    )
                    await self._graph.create_entity(project_entity)
                    await self._activation.record_access(
                        project_entity_id,
                        now,
                        group_id=group_id,
                    )

        layer1_entities: list[dict] = []
        layer1_facts: list[str] = []
        if self._cfg.identity_core_enabled and hasattr(self._graph, "get_identity_core_entities"):
            identity_started = time.perf_counter()
            try:
                core_entities = await self._graph.get_identity_core_entities(group_id)
                for core_entity in core_entities[:identity_limit]:
                    entity_data = await self.entity_to_context_data_bounded(
                        core_entity.id,
                        core_entity.name,
                        core_entity.entity_type,
                        core_entity.summary or "",
                        group_id,
                        now,
                        detail_level="full",
                        timeout_seconds=entity_context_timeout,
                    )
                    layer1_entities.append(entity_data)
                    layer1_facts.extend(entity_data["facts"])
                    seen_ids.add(core_entity.id)
            except Exception:
                logger.debug("Identity core lookup failed (non-fatal)", exc_info=True)
            stage_timings_ms["identity_core"] = _elapsed_ms(identity_started)
        layer1_entities.sort(key=lambda item: item["activation"], reverse=True)
        layer1_text = self.render_tier("## Identity", layer1_entities, layer1_facts)

        layer2_entities: list[dict] = []
        layer2_facts: list[str] = []
        if topic_hint:
            topic_recall_started = time.perf_counter()
            try:
                results = await asyncio.wait_for(
                    self._recall(query=topic_hint, group_id=group_id, limit=project_recall_limit),
                    timeout=project_recall_timeout,
                )
            except TimeoutError:
                logger.debug(
                    "Context topic recall timed out for hint=%r; continuing without layer 2 recall",
                    topic_hint,
                )
                results = []
            stage_timings_ms["topic_recall"] = _elapsed_ms(topic_recall_started)
            for result in results:
                if result.get("result_type") in {"episode", "cue_episode"}:
                    continue
                entity = result.get("entity")
                if not entity or entity["id"] in seen_ids:
                    continue
                hop = result.get("score_breakdown", {}).get("hop_distance")
                if hop is None or hop == 0:
                    detail = "full"
                elif hop == 1:
                    detail = "summary"
                else:
                    detail = "mention"
                entity_data = await self.entity_to_context_data_bounded(
                    entity["id"],
                    entity["name"],
                    entity["type"],
                    entity.get("summary") or "",
                    group_id,
                    now,
                    detail_level=detail,
                    timeout_seconds=entity_context_timeout,
                )
                layer2_entities.append(entity_data)
                layer2_facts.extend(entity_data["facts"])
                seen_ids.add(entity["id"])

        if project_entity_id:
            neighbor_started = time.perf_counter()
            try:
                neighbors = await asyncio.wait_for(
                    self._graph.get_neighbors(
                        project_entity_id,
                        hops=1,
                        group_id=group_id,
                    ),
                    timeout=project_neighbor_timeout,
                )
                for neighbor_entity, _relationship in neighbors[:project_neighbor_limit]:
                    if neighbor_entity.id in seen_ids:
                        continue
                    entity_data = await self.entity_to_context_data_bounded(
                        neighbor_entity.id,
                        neighbor_entity.name,
                        neighbor_entity.entity_type,
                        neighbor_entity.summary or "",
                        group_id,
                        now,
                        detail_level="summary",
                        timeout_seconds=entity_context_timeout,
                    )
                    layer2_entities.append(entity_data)
                    layer2_facts.extend(entity_data["facts"])
                    seen_ids.add(neighbor_entity.id)
            except Exception:
                logger.debug("Project neighbor injection failed (non-fatal)", exc_info=True)
            stage_timings_ms["project_neighbors"] = _elapsed_ms(neighbor_started)

        if project_path and len(layer2_entities) < project_artifact_limit:
            artifact_started = time.perf_counter()
            try:
                artifact_entities = await asyncio.wait_for(
                    self._project_artifact_context_entities(
                        group_id=group_id,
                        project_path=project_path,
                        topic_hint=topic_hint,
                        limit=project_artifact_limit - len(layer2_entities),
                        now=now,
                        seen_ids=seen_ids,
                    ),
                    timeout=min(
                        self._CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS,
                        self._budgeted_timeout(max_tokens, default=1.0),
                    ),
                )
                for entity_data in artifact_entities:
                    layer2_entities.append(entity_data)
                    layer2_facts.extend(entity_data["facts"])
                    seen_ids.add(entity_data["id"])
            except Exception:
                logger.debug("Project artifact context injection failed (non-fatal)", exc_info=True)
            stage_timings_ms["project_artifacts"] = _elapsed_ms(artifact_started)

        if layer2_entities:
            layer2_entities.sort(key=lambda item: item["activation"], reverse=True)
            layer2_text = self.render_tier(
                f"## Project Context ({topic_hint})",
                layer2_entities,
                layer2_facts,
            )
        else:
            layer2_text = ""

        layer3_entities: list[dict] = []
        layer3_facts: list[str] = []
        recency_started = time.perf_counter()
        top = await self._activation.get_top_activated(group_id=group_id, limit=recency_limit)
        stage_timings_ms["recency_lookup"] = _elapsed_ms(recency_started)
        for entity_id, state in top:
            if entity_id in seen_ids:
                continue
            entity = await self._graph.get_entity(entity_id, group_id)
            if not entity:
                continue
            activation = compute_activation(state.access_history, now, self._cfg)
            entity_data = await self.entity_to_context_data_bounded(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                group_id,
                now,
                detail_level="summary",
                timeout_seconds=entity_context_timeout,
            )
            entity_data["activation"] = activation
            layer3_entities.append(entity_data)
            layer3_facts.extend(entity_data["facts"])
            seen_ids.add(entity_id)

        layer3_entities.sort(key=lambda item: item["activation"], reverse=True)
        layer3_text = self.render_tier("## Recent Activity", layer3_entities, layer3_facts)

        layer4_text = ""
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            intentions_started = time.perf_counter()
            try:
                from engram.models.prospective import IntentionMeta

                intention_entities = await self._list_intentions(group_id)
                intention_lines: list[str] = []
                for intention_entity in intention_entities:
                    attrs = intention_entity.attributes or {}
                    try:
                        meta = IntentionMeta(**attrs)
                    except Exception:
                        continue

                    intention_state: ActivationState | None = await self._activation.get_activation(
                        intention_entity.id
                    )
                    activation = 0.0
                    if intention_state:
                        activation = compute_activation(
                            intention_state.access_history,
                            now,
                            self._cfg,
                        )
                    if meta.activation_threshold > 0:
                        warmth_ratio = activation / meta.activation_threshold
                    else:
                        warmth_ratio = 0.0
                    levels = self._cfg.prospective_warmth_levels
                    if warmth_ratio < levels[0]:
                        continue

                    if warmth_ratio >= 1.0:
                        label = "HOT"
                    elif warmth_ratio >= levels[2]:
                        label = "warm"
                    elif warmth_ratio >= levels[1]:
                        label = "warming"
                    else:
                        label = "cool"

                    intention_lines.append(
                        f"- [{label}] {meta.trigger_text} → {meta.action_text} "
                        f"(fires: {meta.fire_count}/{meta.max_fires})"
                    )
                    seen_ids.add(intention_entity.id)

                if intention_lines:
                    layer4_text = "## Active Intentions\n\n" + "\n".join(intention_lines)
            except Exception:
                logger.debug("Intention tier in get_context failed (non-fatal)", exc_info=True)
            stage_timings_ms["intentions"] = _elapsed_ms(intentions_started)

        layer5_text = ""
        pinned_contexts: list[dict] = []
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            pinned_started = time.perf_counter()
            try:
                from engram.models.prospective import IntentionMeta

                pinned_entities = await self._list_intentions(group_id, enabled_only=True)
                pinned_lines: list[str] = []
                for pinned_entity in pinned_entities:
                    attrs = pinned_entity.attributes or {}
                    try:
                        pinned_meta = IntentionMeta(**attrs)
                    except Exception:
                        continue
                    if (
                        pinned_meta.trigger_type != "refresh_context"
                        or not pinned_meta.pinned_result
                    ):
                        continue
                    pinned_contexts.append(
                        {
                            "topic": pinned_meta.trigger_text,
                            "result": pinned_meta.pinned_result,
                            "last_refreshed": pinned_meta.last_refreshed,
                        }
                    )
                    pinned_lines.append(
                        f"### {pinned_meta.trigger_text}\n{pinned_meta.pinned_result}"
                    )
                if pinned_lines:
                    layer5_text = "## Pinned Contexts\n\n" + "\n\n".join(pinned_lines)
            except Exception:
                logger.debug("Pinned context tier in get_context failed (non-fatal)", exc_info=True)
            stage_timings_ms["pinned_contexts"] = _elapsed_ms(pinned_started)

        all_entities = layer1_entities + layer2_entities + layer3_entities
        all_facts = layer1_facts + layer2_facts + layer3_facts
        seen_facts: set[str] = set()
        unique_facts: list[str] = []
        for fact in all_facts:
            if fact in seen_facts:
                continue
            seen_facts.add(fact)
            unique_facts.append(fact)

        sections = [
            section
            for section in [
                cached_packet_text,
                layer1_text,
                layer2_text,
                layer3_text,
                layer4_text,
                layer5_text,
            ]
            if section
        ]
        context_text = (
            "\n\n".join(sections) if sections else "## Active Memory Context\n\nNo memories loaded."
        )

        token_estimate = self.estimate_tokens(context_text)
        if token_estimate > max_tokens:
            char_budget = max_tokens * 4
            context_text = context_text[:char_budget]
            token_estimate = max_tokens

        for entity_data in all_entities:
            await self._activation.record_access(entity_data["id"], now, group_id=group_id)
            await self._publish_access_event(
                entity_data["id"],
                entity_data["name"],
                entity_data["type"],
                group_id,
                "context",
            )

        cache_write_started = time.perf_counter()
        self._cache_context_packets(
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            identity_entities=layer1_entities,
            project_entities=layer2_entities,
        )
        stage_timings_ms["cache_write"] = _elapsed_ms(cache_write_started)
        packet_cache = {
            "hit": bool(cached_packets),
            "packet_count": len(cached_packets),
            "scopes": _packet_scope_counts(cached_packets),
        }

        if format == "briefing" and self._cfg.briefing_enabled:
            # Render a briefing whenever there is renderable content — either
            # activated-entity layers or cached packet text.  Previously a
            # briefing request with no activated entities silently returned a
            # "structured" result, hiding the degradation from callers.
            if all_entities or cached_packet_text:
                growth_stats = await self._collect_growth_stats(group_id)
                briefing = self.template_briefing(
                    context_text,
                    group_id,
                    topic_hint,
                    growth_stats=growth_stats,
                )
                result = {
                    "context": briefing,
                    "entity_count": len(all_entities),
                    "fact_count": len(unique_facts),
                    "token_estimate": self.estimate_tokens(briefing),
                    "format": "briefing",
                    "cached_packets": cached_packets,
                    "packet_cache": packet_cache,
                    "diagnostics": {"stage_timings_ms": dict(stage_timings_ms)},
                }
                if not all_entities:
                    # Briefing built only from cached packets — flag the
                    # degraded source so callers can tell.
                    result["briefing_degraded"] = True
                    result["briefing_degraded_reason"] = "no_activated_entities_cached_only"
                if pinned_contexts:
                    result["pinned_contexts"] = pinned_contexts
                return result

            # Nothing to brief on: fall through to structured output but stamp
            # the degradation explicitly instead of silently swapping format.
            result = {
                "context": context_text,
                "entity_count": len(all_entities),
                "fact_count": len(unique_facts),
                "token_estimate": token_estimate,
                "format": "structured",
                "briefing_degraded": True,
                "briefing_degraded_reason": "no_briefable_content",
                "cached_packets": cached_packets,
                "packet_cache": packet_cache,
                "diagnostics": {"stage_timings_ms": dict(stage_timings_ms)},
            }
            if pinned_contexts:
                result["pinned_contexts"] = pinned_contexts
            return result

        result = {
            "context": context_text,
            "entity_count": len(all_entities),
            "fact_count": len(unique_facts),
            "token_estimate": token_estimate,
            "format": "structured",
            "cached_packets": cached_packets,
            "packet_cache": packet_cache,
            "diagnostics": {"stage_timings_ms": dict(stage_timings_ms)},
        }
        if pinned_contexts:
            result["pinned_contexts"] = pinned_contexts
        return result

    async def _project_artifact_context_entities(
        self,
        *,
        group_id: str,
        project_path: str,
        topic_hint: str | None,
        limit: int,
        now: float,
        seen_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Load a small artifact-backed project tier without triggering bootstrap."""
        if limit <= 0:
            return []
        try:
            artifacts = await self._graph.find_entities(
                entity_type="Artifact",
                group_id=group_id,
                limit=max(limit * 20, 100),
            )
        except Exception:
            logger.debug("Project artifact lookup failed", exc_info=True)
            return []

        project_dir = Path(project_path).expanduser()
        normalized_project = str(project_dir)
        scored: list[tuple[float, int, Entity]] = []
        for artifact in artifacts:
            if artifact.id in seen_ids:
                continue
            attrs = artifact.attributes or {}
            if attrs.get("project_path") != normalized_project:
                continue
            score = _artifact_context_score(topic_hint, artifact, project_path=project_path)
            priority = _artifact_context_priority(str(attrs.get("rel_path") or artifact.name))
            scored.append((score, priority, artifact))
        scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)

        context_entities: list[dict[str, Any]] = []
        for score, _priority, artifact in scored[:limit]:
            attrs = artifact.attributes or {}
            state = await self._activation.get_activation(artifact.id)
            activation = 0.0
            if state:
                from engram.activation.engine import compute_activation

                activation = compute_activation(state.access_history, now, self._cfg)
            context_entities.append(
                {
                    "name": artifact.name,
                    "type": artifact.entity_type,
                    "detail_level": "summary",
                    "id": artifact.id,
                    "activation": activation + min(score, 5.0) / 100.0,
                    "summary": artifact.summary or attrs.get("snippet") or "",
                    "facts": _artifact_context_facts(attrs),
                    "attributes": None,
                }
            )
        return context_entities

    async def entity_to_context_data_bounded(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        now: float,
        *,
        detail_level: str,
        timeout_seconds: float,
    ) -> dict:
        """Build entity context, falling back to summary when graph enrichment is slow."""
        try:
            return await asyncio.wait_for(
                self.entity_to_context_data(
                    entity_id,
                    name,
                    entity_type,
                    summary,
                    group_id,
                    now,
                    detail_level=detail_level,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.debug(
                "Entity context enrichment timed out for entity_id=%s detail=%s",
                entity_id,
                detail_level,
            )
            return {
                "name": name,
                "type": entity_type,
                "detail_level": "summary" if detail_level != "mention" else "mention",
                "id": entity_id,
                "activation": 0.0,
                "summary": summary,
                "facts": [],
                "attributes": None,
            }

    @staticmethod
    def render_cached_packets(packets: Sequence[Mapping[str, Any]]) -> str:
        """Render cached packets as a compact context tier."""
        if not packets:
            return ""
        lines = ["## Cached Memory Packets", ""]
        for packet in packets[:5]:
            trust = packet.get("trust") if isinstance(packet.get("trust"), Mapping) else {}
            source = trust.get("source") or packet.get("source") or "cache"
            freshness = trust.get("freshness") or "unknown"
            title = _redact_packet_text(
                str(packet.get("title") or packet.get("packet_type") or "Memory")
            )
            summary = _redact_packet_text(str(packet.get("summary") or "").strip())
            if summary:
                lines.append(
                    f"- [{source}/{freshness}] {title} — {_truncate_packet_text(summary, 180)}"
                )
            else:
                lines.append(f"- [{source}/{freshness}] {title}")
            why_now = trust.get("why_now") or packet.get("why_now") or packet.get("whyNow")
            if why_now:
                lines.append(f"  - why: {_redact_packet_text(str(why_now))[:160]}")
        return "\n".join(lines)

    def _load_cached_context_packets(
        self,
        *,
        group_id: str,
        topic_hint: str | None,
        project_path: str | None,
    ) -> list[dict[str, Any]]:
        if not self._cfg.recall_packet_cache_enabled or self._get_cached_packets is None:
            return []
        packets: list[dict[str, Any]] = []
        seen_packets: set[tuple[str, str, str]] = set()
        for scope, scope_topic, scope_project in _context_cache_lookup_keys(
            topic_hint=topic_hint,
            project_path=project_path,
        ):
            try:
                hit = self._get_cached_packets(
                    group_id,
                    scope=scope,
                    topic_hint=scope_topic,
                    project_path=scope_project,
                )
            except Exception:
                logger.debug("Context packet-cache lookup failed", exc_info=True)
                continue
            if inspect.isawaitable(hit):
                _close_awaitable(hit)
                continue
            hit_packets = getattr(hit, "packets", None)
            if not isinstance(hit_packets, list):
                continue
            for packet in hit_packets:
                if isinstance(packet, Mapping):
                    _append_unique_cached_packet(
                        packets,
                        seen_packets,
                        _redact_packet_payload(
                            {
                                **dict(packet),
                                "_cache_scope": scope,
                                "_cache_topic_hint": scope_topic,
                                "_cache_project_path": scope_project,
                            }
                        ),
                    )
        return packets

    def _cache_context_packets(
        self,
        *,
        group_id: str,
        topic_hint: str | None,
        project_path: str | None,
        identity_entities: Sequence[Mapping[str, Any]],
        project_entities: Sequence[Mapping[str, Any]],
    ) -> None:
        if not self._cfg.recall_packet_cache_enabled or self._cache_packets is None:
            return
        self._cache_context_scope(
            group_id=group_id,
            scope="identity_core",
            packets=[
                _context_entity_packet(
                    entity,
                    packet_type="identity_core",
                    why_now="Stable identity and preference context for this agent session.",
                )
                for entity in identity_entities
            ],
        )
        if topic_hint or project_path:
            project_packets = [
                _context_entity_packet(
                    entity,
                    packet_type="project_home",
                    why_now="Cached project context for the current workspace.",
                )
                for entity in project_entities
            ]
            self._cache_context_scope(
                group_id=group_id,
                scope="project_home",
                topic_hint=topic_hint,
                project_path=project_path,
                packets=project_packets,
            )
            stable_topic = _derive_context_topic_hint(None, project_path)
            if stable_topic and stable_topic != topic_hint:
                self._cache_context_scope(
                    group_id=group_id,
                    scope="project_home",
                    topic_hint=stable_topic,
                    project_path=project_path,
                    packets=project_packets,
                )

    def _cache_context_scope(
        self,
        *,
        group_id: str,
        scope: str,
        packets: Sequence[Mapping[str, Any]],
        topic_hint: str | None = None,
        project_path: str | None = None,
    ) -> None:
        if not packets or self._cache_packets is None:
            return
        try:
            result = self._cache_packets(
                group_id,
                scope=scope,
                topic_hint=topic_hint,
                project_path=project_path,
                packets=packets,
            )
        except Exception:
            logger.debug("Context packet-cache write failed", exc_info=True)
            return
        if inspect.isawaitable(result):
            _close_awaitable(result)

    @staticmethod
    def _budgeted_entity_limit(
        max_tokens: int,
        *,
        default: int,
        tokens_per_entity: int,
        minimum: int,
    ) -> int:
        budget_limit = max(minimum, max_tokens // tokens_per_entity)
        return max(1, min(default, budget_limit))

    @staticmethod
    def _budgeted_timeout(max_tokens: int, *, default: float) -> float:
        return min(default, max(0.25, max_tokens / 2000.0))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _context_fast_preflight_timeout_seconds(cfg: Any) -> float:
    raw_timeout_ms = getattr(cfg, "context_fast_preflight_timeout_ms", None)
    if raw_timeout_ms is None:
        raw_timeout_ms = getattr(cfg, "recall_fast_preflight_timeout_ms", None)
    if raw_timeout_ms is None:
        raw_timeout_ms = getattr(cfg, "recall_fast_fallback_timeout_ms", 100)
    try:
        timeout_ms = int(raw_timeout_ms)
    except (TypeError, ValueError):
        timeout_ms = 100
    return max(0, timeout_ms) / 1000.0


def _context_fast_preflight_soft_wait_seconds(cfg: Any) -> float:
    timeout_seconds = _context_fast_preflight_timeout_seconds(cfg)
    if timeout_seconds <= 0:
        return 0.0
    raw_soft_wait_ms = getattr(cfg, "context_fast_preflight_soft_wait_ms", None)
    if raw_soft_wait_ms is None:
        soft_wait_ms = 75
    else:
        try:
            soft_wait_ms = int(raw_soft_wait_ms)
        except (TypeError, ValueError):
            soft_wait_ms = 75
    return min(timeout_seconds, max(0, soft_wait_ms) / 1000.0)


async def _await_loaded_store_context_preflight_task(
    task: asyncio.Task[dict[str, Any] | None],
    *,
    project_file_task: asyncio.Task[tuple[list[dict[str, Any]], float]] | None,
    soft_wait_seconds: float,
) -> dict[str, Any] | None:
    if soft_wait_seconds <= 0:
        return None
    try:
        done, _pending = await asyncio.wait({task}, timeout=soft_wait_seconds)
        if task in done:
            return task.result()
        if project_file_task is None:
            return await task
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Context loaded-store preflight task failed", exc_info=True)
        return None


def _consume_loaded_store_context_task(
    task: asyncio.Task[dict[str, Any] | None],
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("Background context loaded-store preflight failed", exc_info=True)


def _context_packet_name_resolver(manager: Any, group_id: str) -> Callable[[str], Awaitable[str]]:
    async def resolve_entity_name(entity_id: str) -> str:
        resolver = getattr(manager, "resolve_entity_name", None)
        if not callable(resolver):
            return entity_id
        value = resolver(entity_id, group_id)
        if inspect.isawaitable(value):
            value = await value
        return str(value or entity_id)

    return resolve_entity_name


def _camel_stage_name(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _derive_context_topic_hint(
    topic_hint: str | None,
    project_path: str | None,
) -> str | None:
    if topic_hint or not project_path:
        return topic_hint
    project_dir = Path(project_path).expanduser()
    if project_dir.name and str(project_dir) != str(Path.home()):
        return project_dir.name
    return topic_hint


def _artifact_context_score(
    topic_hint: str | None,
    artifact: Entity,
    *,
    project_path: str | None,
) -> float:
    attrs = artifact.attributes or {}
    project_terms: set[str] = set()
    if project_path:
        project_name = Path(project_path).expanduser().name.lower()
        if project_name:
            project_terms.add(project_name)
    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (topic_hint or "").lower())
        if term
        not in {
            "and",
            "for",
            "from",
            "the",
            "this",
            "that",
            "with",
            "project",
            "context",
        }
        and term not in project_terms
    }
    text_parts = [
        artifact.name,
        artifact.summary or "",
        str(attrs.get("rel_path") or ""),
        str(attrs.get("snippet") or ""),
    ]
    claims = attrs.get("claims") if isinstance(attrs.get("claims"), list) else []
    for claim in claims:
        if isinstance(claim, Mapping):
            text_parts.extend(
                str(claim.get(key) or "") for key in ("subject", "predicate", "object")
            )
    text = " ".join(text_parts).lower()
    if not terms:
        return 0.0
    rel_path = str(attrs.get("rel_path") or artifact.name).lower()
    return float(
        sum(text.count(term) for term in terms) + sum(3 for term in terms if term in rel_path)
    )


def _artifact_context_priority(rel_path: str) -> int:
    lowered = rel_path.lower()
    if lowered == "docs/current_handoff.md":
        return 120
    if lowered == "readme.md":
        return 100
    if lowered in {"pyproject.toml", "package.json", "makefile", ".env.example"}:
        return 90
    if lowered.startswith("docs/"):
        return 80
    if lowered.startswith("notes/") or lowered.startswith("memory/"):
        return 70
    return 10


def _artifact_context_facts(attrs: Mapping[str, Any]) -> list[str]:
    claims = attrs.get("claims") if isinstance(attrs.get("claims"), list) else []
    facts: list[str] = []
    for claim in claims[:3]:
        if not isinstance(claim, Mapping):
            continue
        predicate = claim.get("predicate")
        obj = claim.get("object")
        if predicate and obj:
            facts.append(f"{predicate}={obj}")
    return facts


def _bounded_context_topic_hint(
    topic_hint: str | None,
    project_path: str | None,
) -> str | None:
    """Prefer stable project context when a long hint already names the project."""
    if not topic_hint:
        return _derive_context_topic_hint(topic_hint, project_path)
    stable_topic = _derive_context_topic_hint(None, project_path)
    if not stable_topic or stable_topic == topic_hint:
        return topic_hint
    hint_text = topic_hint.lower()
    stable_text = stable_topic.lower()
    if stable_text in hint_text and len(topic_hint.split()) > 3:
        return stable_topic
    return topic_hint


def _context_cache_lookup_keys(
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> list[tuple[str, str | None, str | None]]:
    keys: list[tuple[str, str | None, str | None]] = [("identity_core", None, None)]
    keys.append((SESSION_RECENT_PACKET_SCOPE, None, None))
    if not (topic_hint or project_path):
        return keys

    stable_topic = _derive_context_topic_hint(None, project_path)
    candidates: list[tuple[str | None, str | None]] = [
        (topic_hint, project_path),
        (stable_topic, project_path),
        (None, project_path),
        (topic_hint, None),
        (stable_topic, None),
    ]
    seen: set[tuple[str | None, str | None]] = set()
    for candidate in candidates:
        if not any(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        keys.append(("project_home", candidate[0], candidate[1]))
    return keys


def _cached_context_payload_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    status: str,
    duration_ms: float,
    skip_reason: str | None,
    timeout: bool,
    allow_session_recent_only: bool = True,
) -> dict[str, Any] | None:
    packets = _load_cached_context_packets_from_manager(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
    )
    packets = _select_relevant_context_packets(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    )
    if not packets:
        return None
    if not allow_session_recent_only and _context_packets_need_loaded_store_enrichment(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        return None
    context = MemoryContextBuilder.render_cached_packets(packets)
    payload: dict[str, Any] = {
        "context": context,
        "entity_count": 0,
        "fact_count": 0,
        "token_estimate": MemoryContextBuilder.estimate_tokens(context),
        "format": "structured" if format == "briefing" else format,
        "cached_packets": packets,
        "packet_cache": {
            "hit": True,
            "packet_count": len(packets),
            "scopes": _packet_scope_counts(packets),
        },
        "status": status,
        "budget": {
            "profile": budget.profile,
            "surface": budget.surface,
            "mode": budget.mode,
            "max_wall_ms": budget.max_wall_ms,
            "duration_ms": duration_ms,
            "budget_miss": bool(timeout or budget.exceeded(duration_ms)),
            "timeout": timeout,
            "degraded": status == "degraded",
            "skip_reason": skip_reason,
        },
        "lifecycle": {
            "stage": "recall",
            "degraded": status == "degraded",
            "timeout": timeout,
            "skip_reason": skip_reason,
        },
        "diagnostics": {
            "stage_timings_ms": {
                "packet_cache": 0.0,
                "cache_fallback": duration_ms,
            }
        },
    }
    if format == "briefing":
        payload["briefing_degraded"] = True
        payload["briefing_degraded_reason"] = "cache_fast_path"
    return payload


async def _durable_context_payload_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    started: float,
) -> dict[str, Any] | None:
    """Build get_context packets from durable graph facts (Decision/Preference/…).

    Reuses explicit-recall durable-entity rescue so session-start continuity
    matches the golden path without requiring the agent to call recall first.

    Hard latency budget: pack build is capped at ~1s wall. Process cache (45s TTL)
    makes repeated session-start get_context sub-millisecond after the first hit.
    """
    from engram.retrieval.recall_surface import _durable_entity_name_rescue

    raw_topic = (topic_hint or "").strip()
    query = raw_topic or _derive_context_topic_hint(None, project_path)
    if not query or query.casefold() in {"engram", "default", "project"}:
        query = _DEFAULT_DURABLE_CONTEXT_QUERY
    elif len(query.split()) <= 3:
        # Short project names: blend durable cues so probes still hit Decisions.
        query = f"{query} decisions preferences goals"
    # Long topic strings (full Decision names) keep their exact text so
    # exact-name rescue can hit without BM25 thrash.

    cfg = _manager_activation_config(manager)
    max_packets = max(1, min(5, int(getattr(cfg, "recall_packet_explicit_limit", 3) or 3)))
    pack_started = time.perf_counter()

    # --- A1: process-level cache (fast path) ---
    cached = _load_durable_context_process_cache(group_id, query)
    if cached is not None:
        packet_payloads, entity_ids = cached
        return _finalize_durable_context_payload(
            packet_payloads,
            entity_ids=entity_ids,
            format=format,
            budget=budget,
            started=started,
            pack_started=pack_started,
            cache_hit=True,
            hard_budget_miss=False,
            timed_out=False,
        )

    hard_budget = _DURABLE_CONTEXT_HARD_BUDGET_SECONDS
    # Per-probe cap is a fraction of the hard wall so rescue+type stay inside budget.
    probe_timeout = min(0.75, max(0.2, hard_budget * 0.4))

    async def _collect_durable_results() -> list[dict[str, Any]]:
        # 1) identity_core list first (product continuity core, usually small/fast)
        results: list[dict[str, Any]] = []
        try:
            typed = await _list_durable_entities_by_type(
                manager,
                group_id=group_id,
                limit=max_packets,
                timeout_seconds=probe_timeout,
            )
            if isinstance(typed, list):
                results.extend(typed)
        except Exception:
            logger.debug("Durable type listing failed", exc_info=True)
        if len(results) >= max_packets:
            return results[:max_packets]
        # 2) name rescue only if type/identity path was thin
        try:
            rescue = await _durable_entity_name_rescue(
                manager,
                group_id=group_id,
                query=query,
                limit=max_packets,
                timeout_seconds=probe_timeout,
            )
            if isinstance(rescue, list):
                seen = {str((r.get("entity") or {}).get("id") or "") for r in results}
                for item in rescue:
                    eid = str((item.get("entity") or {}).get("id") or "")
                    if eid and eid not in seen:
                        results.append(item)
                        seen.add(eid)
                    if len(results) >= max_packets:
                        break
        except Exception:
            logger.debug("Durable context name rescue failed", exc_info=True)
        return results

    timed_out = False
    hard_budget_miss = False
    results: list[dict[str, Any]] = []
    try:
        results = await asyncio.wait_for(
            _collect_durable_results(),
            timeout=hard_budget,
        )
    except TimeoutError:
        timed_out = True
        hard_budget_miss = True
        logger.debug(
            "Durable context pack hit hard %.2fs budget (group=%s)",
            hard_budget,
            group_id,
        )
        results = []
    except Exception:
        logger.debug("Durable context pack failed", exc_info=True)
        results = []

    pack_elapsed = time.perf_counter() - pack_started
    if pack_elapsed > hard_budget:
        hard_budget_miss = True

    if not results:
        return None

    remaining = max(0.05, hard_budget - (time.perf_counter() - pack_started))
    try:
        packets = await asyncio.wait_for(
            assemble_memory_packets(
                list(results),
                query,
                mode="context_durable",
                max_packets=max_packets,
                resolve_entity_name=_context_packet_name_resolver(manager, group_id),
            ),
            timeout=remaining,
        )
    except TimeoutError:
        timed_out = True
        hard_budget_miss = True
        return None
    except Exception:
        logger.debug("Durable context packet assembly failed", exc_info=True)
        return None
    if not packets:
        return None

    packet_payloads = [
        _redact_packet_payload({**packet.to_dict(), "_cache_scope": DURABLE_CONTEXT_PACKET_SCOPE})
        for packet in packets
    ]
    entity_ids: set[str] = set()
    for packet in packet_payloads:
        for key in ("entity_ids", "entityIds"):
            raw = packet.get(key) or []
            if isinstance(raw, list | tuple):
                entity_ids.update(str(item) for item in raw if item)

    _store_durable_context_process_cache(
        group_id,
        query,
        packet_payloads,
        entity_ids,
    )
    _cache_durable_context_packets(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        packets=packet_payloads,
        build_duration_ms=_elapsed_ms(pack_started),
    )

    return _finalize_durable_context_payload(
        packet_payloads,
        entity_ids=entity_ids,
        format=format,
        budget=budget,
        started=started,
        pack_started=pack_started,
        cache_hit=False,
        hard_budget_miss=hard_budget_miss,
        timed_out=timed_out,
    )


def _finalize_durable_context_payload(
    packet_payloads: Sequence[Mapping[str, Any]],
    *,
    entity_ids: set[str],
    format: str,
    budget: RecallBudget,
    started: float,
    pack_started: float,
    cache_hit: bool,
    hard_budget_miss: bool,
    timed_out: bool,
) -> dict[str, Any] | None:
    context = MemoryContextBuilder.render_cached_packets(packet_payloads)
    if not context.strip():
        return None
    duration_ms = _elapsed_ms(started)
    pack_ms = _elapsed_ms(pack_started)
    # Hard product budget is 1s for the durable pack itself; also honor surface budget.
    budget_miss = bool(hard_budget_miss or budget.exceeded(duration_ms))
    payload: dict[str, Any] = {
        "context": context,
        "entity_count": len(entity_ids),
        "fact_count": len(packet_payloads),
        "token_estimate": MemoryContextBuilder.estimate_tokens(context),
        "format": "structured" if format == "briefing" else format,
        "cached_packets": list(packet_payloads),
        "packet_cache": {
            "hit": cache_hit,
            "packet_count": len(packet_payloads),
            "scopes": _packet_scope_counts(packet_payloads),
        },
        "status": "ok",
        "budget": {
            "profile": budget.profile,
            "surface": budget.surface,
            "mode": budget.mode,
            "max_wall_ms": budget.max_wall_ms,
            "duration_ms": duration_ms,
            "budget_miss": budget_miss,
            "timeout": timed_out,
            "degraded": hard_budget_miss or timed_out,
            "skip_reason": "durable_context_hard_budget" if hard_budget_miss else None,
        },
        "lifecycle": {
            "stage": "recall",
            "degraded": hard_budget_miss or timed_out,
            "timeout": timed_out,
            "skip_reason": "durable_context_hard_budget" if hard_budget_miss else None,
            "fallback_status": "durable_context",
        },
        "diagnostics": {
            "stage_timings_ms": {
                "durable_context_pack": pack_ms,
                "durable_context_cache_hit": 1.0 if cache_hit else 0.0,
            },
        },
    }
    if format == "briefing":
        payload["format"] = "briefing"
        payload["context"] = _render_durable_briefing(packet_payloads)
        payload["token_estimate"] = MemoryContextBuilder.estimate_tokens(str(payload["context"]))
    return payload


def _cache_durable_context_packets(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    packets: Sequence[Mapping[str, Any]],
    build_duration_ms: float = 0.0,
) -> None:
    """Mirror durable packs into the manager packet cache when enabled."""
    cache_packets = getattr(manager, "cache_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if (
        not getattr(cfg, "recall_packet_cache_enabled", False)
        or not callable(cache_packets)
        or not packets
    ):
        return
    packets_to_cache = [
        {key: value for key, value in dict(packet).items() if key != "_cache_scope"}
        for packet in packets
    ]
    try:
        result = cache_packets(
            group_id,
            scope=DURABLE_CONTEXT_PACKET_SCOPE,
            topic_hint=topic_hint,
            project_path=project_path,
            packets=packets_to_cache,
            build_duration_ms=build_duration_ms,
            persist=False,
        )
    except Exception:
        logger.debug("Durable context packet-cache write failed", exc_info=True)
        return
    if inspect.isawaitable(result):
        _close_awaitable(result)


def _render_durable_briefing(packets: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic 2–3 sentence briefing from durable graph packets (no LLM)."""
    facts: list[str] = []
    for packet in list(packets)[:3]:
        title = str(packet.get("title") or "").removeprefix("Fact: ").strip()
        summary = str(packet.get("summary") or "").strip()
        if title and summary and summary.casefold() not in title.casefold():
            facts.append(f"{title} — {summary}")
        elif title:
            facts.append(title)
        elif summary:
            facts.append(summary)
    if not facts:
        return ""
    if len(facts) == 1:
        return f"Key memory to carry forward: {facts[0]}."
    numbered = " ".join(f"({i}) {fact}." for i, fact in enumerate(facts, start=1))
    return f"Key memories to carry forward: {numbered}"


async def _list_durable_entities_by_type(
    manager: Any,
    *,
    group_id: str,
    limit: int = 5,
    timeout_seconds: float = 1.5,
) -> list[dict[str, Any]]:
    """List Decision/Preference/… entities, dropping decision_statement scrap.

    Prefer graph ``find_entities_by_type`` / identity_core listing over hybrid
    search — search thrash on large native brains regularly blows the 1s
    durable-context budget and leaves get_context empty.
    """
    from engram.extraction.promotion import durable_result_boost, is_durable_recall_entity_type
    from engram.retrieval.recall_surface import _is_decision_statement_noise

    graph = getattr(manager, "_graph", None) or getattr(manager, "graph_store", None)
    find_by_type = getattr(graph, "find_entities_by_type", None) if graph is not None else None
    get_identity = getattr(graph, "get_identity_core_entities", None) if graph is not None else None
    search = getattr(manager, "search_entities", None)
    if not callable(find_by_type) and not callable(search) and not callable(get_identity):
        return []

    types = ("Decision", "Preference", "Goal", "Commitment", "Correction", "Person")

    async def _fetch_type(entity_type: str) -> list[Any]:
        try:
            if callable(find_by_type):
                value = find_by_type(entity_type, group_id, limit=max(limit, 8))
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=timeout_seconds)
                if isinstance(value, list) and value:
                    return value
            if callable(search):
                value = search(
                    group_id=group_id,
                    entity_type=entity_type,
                    limit=max(limit, 8),
                )
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=timeout_seconds)
                return value if isinstance(value, list) else []
        except Exception:
            return []
        return []

    hits: list[dict[str, Any]] = []
    # Identity core first (usually small and continuity-critical).
    if callable(get_identity):
        try:
            core = get_identity(group_id)
            if inspect.isawaitable(core):
                core = await asyncio.wait_for(core, timeout=min(timeout_seconds, 0.5))
            for ent in core or []:
                if isinstance(ent, Mapping):
                    name = str(ent.get("name") or "")
                    et = str(ent.get("entity_type") or ent.get("type") or "")
                    eid = str(ent.get("id") or "")
                    summary = str(ent.get("summary") or "")
                else:
                    name = str(getattr(ent, "name", "") or "")
                    et = str(getattr(ent, "entity_type", "") or "")
                    eid = str(getattr(ent, "id", "") or "")
                    summary = str(getattr(ent, "summary", "") or "")
                if not eid or not name or _is_decision_statement_noise(name):
                    continue
                if not is_durable_recall_entity_type(et):
                    continue
                hits.append(
                    {
                        "result_type": "entity",
                        "score": 0.95,
                        "entity": {
                            "id": eid,
                            "name": name,
                            "type": et,
                            "summary": summary,
                        },
                        "relationships": [],
                        "score_breakdown": {"source": "identity_core_list"},
                        "source": "durable_type_list",
                    }
                )
                if len(hits) >= limit:
                    return hits[:limit]
        except Exception:
            logger.debug("identity_core list for durable context failed", exc_info=True)

    # Parallel type probes keep session-start get_context under ~1s on loaded brains.
    typed_rows = await asyncio.gather(*(_fetch_type(entity_type) for entity_type in types))
    for entity_type, value in zip(types, typed_rows):
        for row in value:
            if not isinstance(row, Mapping) and not hasattr(row, "name"):
                continue
            if isinstance(row, Mapping):
                name = str(row.get("name") or "")
                et = str(row.get("entity_type") or row.get("type") or entity_type)
            else:
                name = str(getattr(row, "name", "") or "")
                et = str(getattr(row, "entity_type", "") or entity_type)
            if not name or _is_decision_statement_noise(name):
                continue
            # Normalize entity rows into Mapping-like for the original loop body.
            if not isinstance(row, Mapping):
                row = {
                    "id": getattr(row, "id", ""),
                    "name": name,
                    "entity_type": et,
                    "summary": getattr(row, "summary", "") or "",
                    "identity_core": bool(getattr(row, "identity_core", False)),
                    "activation_score": 0.0,
                }
            et = str(row.get("entity_type") or row.get("type") or entity_type)
            if not is_durable_recall_entity_type(et):
                continue
            eid = str(row.get("id") or "")
            if not eid:
                continue
            activation = float(row.get("activation_score") or 0.0)
            score = 0.55 + durable_result_boost(et) * 0.08 + min(0.2, activation * 0.2)
            # Prefer identity-core promoted facts when the store exposes the flag.
            if row.get("identity_core") in (True, 1, "1"):
                score += 0.1
            hits.append(
                {
                    "result_type": "entity",
                    "score": min(0.95, score),
                    "entity": {
                        "id": eid,
                        "name": name,
                        "type": et,
                        "summary": str(row.get("summary") or ""),
                    },
                    "relationships": [],
                    "score_breakdown": {
                        "relevance_confidence": min(0.95, score),
                        "planner_support": 0.0,
                        "rescue": "durable_type_list",
                    },
                    "source": "durable_context_type_list",
                }
            )
    hits.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return hits[: max(1, limit)]


async def _loaded_store_context_payload_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    started: float,
) -> dict[str, Any] | None:
    """Build topic context from the bounded cue/episode store before file fallback."""
    query = topic_hint or _derive_context_topic_hint(None, project_path)
    if not query:
        return None
    fallback = getattr(manager, "fast_recall_fallback", None)
    if not callable(fallback):
        return None

    cfg = _manager_activation_config(manager)
    max_packets = max(1, int(getattr(cfg, "recall_packet_explicit_limit", 3) or 3))
    timeout_seconds = _context_fast_preflight_timeout_seconds(cfg)
    if timeout_seconds <= 0:
        return None

    preflight_started = time.perf_counter()
    search_started = time.perf_counter()
    try:
        value = fallback(query=query, group_id=group_id, limit=max_packets * 2)
        if inspect.isawaitable(value):
            value = await asyncio.wait_for(value, timeout=timeout_seconds)
    except TimeoutError:
        return None
    except Exception:
        logger.debug("Context loaded-store preflight failed", exc_info=True)
        return None
    search_duration_ms = _elapsed_ms(search_started)
    if not isinstance(value, list) or not value:
        return None
    value = _prefer_project_context_results(value, project_path=project_path)

    assembly_started = time.perf_counter()
    packets = await assemble_memory_packets(
        list(value),
        query,
        mode="context_preflight",
        max_packets=max_packets,
        resolve_entity_name=_context_packet_name_resolver(manager, group_id),
    )
    assembly_duration_ms = _elapsed_ms(assembly_started)
    packet_payloads = [
        _redact_packet_payload(
            {**packet.to_dict(), "_cache_scope": LOADED_STORE_CONTEXT_PACKET_SCOPE}
        )
        for packet in packets
    ]
    if not packet_payloads or not _context_cached_packets_relevant(
        packet_payloads,
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        return None

    _cache_loaded_store_context_packets(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        packets=packet_payloads,
    )
    context = MemoryContextBuilder.render_cached_packets(packet_payloads)
    duration_ms = _elapsed_ms(started)
    payload: dict[str, Any] = {
        "context": context,
        "entity_count": 0,
        "fact_count": 0,
        "token_estimate": MemoryContextBuilder.estimate_tokens(context),
        "format": "structured" if format == "briefing" else format,
        "cached_packets": packet_payloads,
        "packet_cache": {
            "hit": False,
            "packet_count": len(packet_payloads),
            "scopes": _packet_scope_counts(packet_payloads),
        },
        "status": "ok",
        "budget": {
            "profile": budget.profile,
            "surface": budget.surface,
            "mode": budget.mode,
            "max_wall_ms": budget.max_wall_ms,
            "duration_ms": duration_ms,
            "budget_miss": budget.exceeded(duration_ms),
            "timeout": False,
            "degraded": False,
            "skip_reason": None,
        },
        "lifecycle": {
            "stage": "recall",
            "degraded": False,
            "timeout": False,
            "skip_reason": None,
        },
        "diagnostics": {
            "stage_timings_ms": {
                "loaded_store_context_preflight": _elapsed_ms(preflight_started),
                "loaded_store_context_search": search_duration_ms,
                "loaded_store_context_packet_assembly": assembly_duration_ms,
            }
        },
    }
    if format == "briefing":
        payload["briefing_degraded"] = True
        payload["briefing_degraded_reason"] = "loaded_store_fast_path"
    return payload


def _project_file_context_payload_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    status: str,
    duration_ms: float,
    skip_reason: str | None,
    timeout: bool,
    pre_fallback_stage_timings: Mapping[str, float | None] | None = None,
) -> dict[str, Any] | None:
    fallback_started = time.perf_counter()
    packets = project_file_fallback_packet_payloads(
        manager,
        group_id=group_id,
        project_path=project_path,
        topic_hint=topic_hint,
        max_packets=5,
        reason=_project_file_fallback_reason(skip_reason),
    )
    if not packets:
        return None
    return _project_file_context_payload_from_packets(
        packets,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
        budget=budget,
        status=status,
        duration_ms=duration_ms,
        fallback_duration_ms=_elapsed_ms(fallback_started),
        skip_reason=skip_reason,
        timeout=timeout,
        pre_fallback_stage_timings=pre_fallback_stage_timings,
    )


async def _project_file_context_payload_from_task_or_manager(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]] | None,
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    status: str,
    duration_ms: float,
    skip_reason: str | None,
    timeout: bool,
    started: float,
    pre_fallback_stage_timings: Mapping[str, float | None] | None = None,
) -> dict[str, Any] | None:
    if task is None:
        return _project_file_context_payload_from_manager(
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            pre_fallback_stage_timings=pre_fallback_stage_timings,
        )
    stage_timings = dict(pre_fallback_stage_timings or {})
    if not task.done():
        rescue_payload = _project_file_context_rescue_payload_from_task(
            task,
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            started=started,
            pre_fallback_stage_timings=stage_timings,
        )
        if rescue_payload is not None:
            return rescue_payload
    soft_wait_started = time.perf_counter()
    if not task.done():
        await _wait_for_project_file_context_task_before_rescue(
            task,
            soft_wait_seconds=_context_fast_preflight_soft_wait_seconds(
                _manager_activation_config(manager)
            ),
        )
    soft_wait_duration_ms = _elapsed_ms(soft_wait_started)
    if soft_wait_duration_ms > 0:
        stage_timings["project_file_fallback_soft_wait"] = soft_wait_duration_ms
    if not task.done():
        cached_packets = _project_file_cache_rescue_packets_from_manager(
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            max_packets=5,
        )
        if cached_packets:
            task.add_done_callback(
                partial(
                    _cache_project_file_context_fallback_task_result,
                    manager=manager,
                    group_id=group_id,
                    topic_hint=topic_hint,
                    project_path=project_path,
                )
            )
            return _project_file_context_payload_from_packets(
                cached_packets,
                group_id=group_id,
                topic_hint=topic_hint,
                project_path=project_path,
                format=format,
                budget=budget,
                status=status,
                duration_ms=duration_ms,
                fallback_duration_ms=0.0,
                skip_reason=skip_reason,
                timeout=timeout,
                total_duration_ms=_elapsed_ms(started),
                pre_fallback_stage_timings={
                    **stage_timings,
                    "project_file_fallback_cache_rescue": 0.0,
                    "project_file_fallback_pending": 1.0,
                },
                cache_hit=True,
            )
        pending_packets = _project_file_pending_context_packets(
            project_path=project_path,
            topic_hint=topic_hint,
            reason=_project_file_fallback_reason(skip_reason),
        )
        if pending_packets:
            task.add_done_callback(
                partial(
                    _cache_project_file_context_fallback_task_result,
                    manager=manager,
                    group_id=group_id,
                    topic_hint=topic_hint,
                    project_path=project_path,
                )
            )
            return _project_file_context_payload_from_packets(
                pending_packets,
                group_id=group_id,
                topic_hint=topic_hint,
                project_path=project_path,
                format=format,
                budget=budget,
                status=status,
                duration_ms=duration_ms,
                fallback_duration_ms=0.0,
                skip_reason=skip_reason,
                timeout=timeout,
                total_duration_ms=_elapsed_ms(started),
                pre_fallback_stage_timings={
                    **stage_timings,
                    "project_file_fallback_pending": 1.0,
                },
                cache_hit=False,
            )
    try:
        packets, fallback_duration_ms = await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Prebuilt project file context fallback failed", exc_info=True)
        return _project_file_context_payload_from_manager(
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            format=format,
            budget=budget,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            pre_fallback_stage_timings=stage_timings,
        )
    if not packets:
        return None
    cache_project_file_fallback_packet_payloads(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        packets=packets,
    )
    return _project_file_context_payload_from_packets(
        packets,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
        budget=budget,
        status=status,
        duration_ms=duration_ms,
        fallback_duration_ms=fallback_duration_ms,
        skip_reason=skip_reason,
        timeout=timeout,
        total_duration_ms=_elapsed_ms(started),
        pre_fallback_stage_timings=stage_timings,
    )


def _project_file_context_rescue_payload_from_task(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]] | None,
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    status: str,
    duration_ms: float,
    skip_reason: str | None,
    timeout: bool,
    started: float,
    pre_fallback_stage_timings: Mapping[str, float | None] | None = None,
) -> dict[str, Any] | None:
    if task is None or task.done():
        return None
    cached_packets = _project_file_cache_rescue_packets_from_manager(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        max_packets=5,
    )
    if not cached_packets:
        return None
    task.add_done_callback(
        partial(
            _cache_project_file_context_fallback_task_result,
            manager=manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
        )
    )
    stage_timings = dict(pre_fallback_stage_timings or {})
    return _project_file_context_payload_from_packets(
        cached_packets,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
        budget=budget,
        status=status,
        duration_ms=duration_ms,
        fallback_duration_ms=0.0,
        skip_reason=skip_reason,
        timeout=timeout,
        total_duration_ms=_elapsed_ms(started),
        pre_fallback_stage_timings={
            **stage_timings,
            "project_file_fallback_soft_wait": 0.0,
            "project_file_fallback_cache_rescue": 0.0,
            "project_file_fallback_pending": 1.0,
        },
        cache_hit=True,
    )


async def _wait_for_project_file_context_task_before_rescue(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]],
    *,
    soft_wait_seconds: float,
) -> None:
    if soft_wait_seconds <= 0:
        return
    try:
        await asyncio.wait({task}, timeout=soft_wait_seconds)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Project file context soft wait failed", exc_info=True)


def _project_file_pending_context_packets(
    *,
    project_path: str | None,
    topic_hint: str | None,
    reason: str,
) -> list[dict[str, Any]]:
    """Return a small useful packet while a cold project-file scan finishes."""
    if not project_path:
        return []
    try:
        project_dir = Path(project_path).expanduser()
        if not project_dir.is_dir() or str(project_dir) in {str(Path.home()), "/"}:
            return []
    except OSError:
        return []

    markers = [
        marker
        for marker in ("README.md", "pyproject.toml", "package.json", "docs")
        if (project_dir / marker).exists()
    ]
    topic_text = f" for `{topic_hint}`" if topic_hint else ""
    marker_text = f" Detected project markers: {', '.join(markers[:4])}." if markers else ""
    summary = (
        f"Project {project_dir.name} at {project_dir} has local file context "
        f"warming in the background{topic_text}.{marker_text}"
    )
    return [
        _redact_packet_payload(
            {
                "packet_type": "project_home",
                "title": f"Project Context Warming: {project_dir.name}",
                "summary": _redact_packet_text(summary),
                "why_now": reason,
                "confidence": 0.45,
                "entity_ids": [],
                "relationship_ids": [],
                "episode_ids": [],
                "evidence_lines": [
                    _redact_packet_text(f"project_path={project_dir}"),
                    _redact_packet_text(
                        "Full project-file packets are being cached in the background."
                    ),
                ],
                "provenance": [f"project:{project_dir.name}"],
                "supporting_intents": ["project_file_context_fallback_pending"],
                "trust": {
                    "freshness": "pending",
                    "source": "project_file",
                    "confidence": 0.45,
                    "why_now": reason,
                    "provenance_count": 1,
                    "evidence_count": 2,
                    "belief_status": "unknown",
                    "confirmed_count": 0,
                    "corrected_count": 0,
                    "dismissed_count": 0,
                    "last_confirmed_at": None,
                    "last_corrected_at": None,
                    "last_dismissed_at": None,
                },
                "_cache_scope": "project_file_pending",
                "_project_file_fallback_pending": True,
            }
        )
    ]


def _project_file_cache_rescue_packets_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    max_packets: int,
) -> list[dict[str, Any]]:
    """Return cached project-file packets for the same project while a fresh scan runs."""
    if not project_path or max_packets <= 0:
        return []
    get_cached_packets = getattr(manager, "get_cached_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if not getattr(cfg, "recall_packet_cache_enabled", False) or not callable(get_cached_packets):
        return []

    stable_topic = _derive_context_topic_hint(None, project_path)
    candidates = [(stable_topic, project_path), (None, project_path)]
    seen_keys: set[tuple[str | None, str | None]] = set()
    seen_packets: set[tuple[str, str, str]] = set()
    packets: list[dict[str, Any]] = []
    for cache_topic, cache_project_path in candidates:
        if len(packets) >= max_packets:
            break
        if not (cache_topic or cache_project_path):
            continue
        key = (cache_topic, cache_project_path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            hit = get_cached_packets(
                group_id,
                scope="project_home",
                topic_hint=cache_topic,
                project_path=cache_project_path,
                sync_persistent=True,
            )
        except Exception:
            logger.debug("Project-file cache rescue lookup failed", exc_info=True)
            continue
        if inspect.isawaitable(hit):
            _close_awaitable(hit)
            continue
        hit_packets = getattr(hit, "packets", None)
        if not isinstance(hit_packets, list):
            continue
        for packet in hit_packets:
            if not isinstance(packet, Mapping):
                continue
            if not _context_packet_is_project_file_cache_rescue_candidate(
                packet,
                project_path=project_path,
                cache_project_path=cache_project_path,
            ):
                continue
            rescue_packet = _redact_packet_payload(
                {
                    **dict(packet),
                    "_cache_scope": "project_file_cache_rescue",
                    "_cache_topic_hint": cache_topic,
                    "_cache_project_path": cache_project_path,
                    "_project_file_fallback_cache_rescue": True,
                }
            )
            _append_unique_cached_packet(packets, seen_packets, rescue_packet)
            if len(packets) >= max_packets:
                break
    return _select_relevant_context_packets(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    )


def _cache_project_file_context_fallback_task_result(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]],
    *,
    manager: Any,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
) -> None:
    try:
        packets, _duration_ms = task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("Project file context fallback background cache failed", exc_info=True)
        return
    if not packets:
        return
    cache_project_file_fallback_packet_payloads(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        packets=packets,
    )


def _project_file_context_payload_from_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    format: str,
    budget: RecallBudget,
    status: str,
    duration_ms: float,
    fallback_duration_ms: float,
    skip_reason: str | None,
    timeout: bool,
    total_duration_ms: float | None = None,
    pre_fallback_stage_timings: Mapping[str, float | None] | None = None,
    cache_hit: bool = False,
) -> dict[str, Any]:
    context = MemoryContextBuilder.render_cached_packets(packets)
    effective_total_duration_ms = round(
        total_duration_ms if total_duration_ms is not None else duration_ms + fallback_duration_ms,
        4,
    )
    payload: dict[str, Any] = {
        "context": context,
        "entity_count": 0,
        "fact_count": 0,
        "token_estimate": MemoryContextBuilder.estimate_tokens(context),
        "format": "structured" if format == "briefing" else format,
        "cached_packets": packets,
        "packet_cache": {
            "hit": cache_hit,
            "packet_count": len(packets),
            "scopes": _packet_scope_counts(packets),
        },
        "status": status,
        "budget": {
            "profile": budget.profile,
            "surface": budget.surface,
            "mode": budget.mode,
            "max_wall_ms": budget.max_wall_ms,
            "duration_ms": effective_total_duration_ms,
            "budget_miss": bool(timeout or budget.exceeded(effective_total_duration_ms)),
            "timeout": timeout,
            "degraded": status == "degraded",
            "skip_reason": skip_reason,
        },
        "lifecycle": {
            "stage": "recall",
            "degraded": status == "degraded",
            "timeout": timeout,
            "skip_reason": skip_reason,
        },
        "diagnostics": {
            "stage_timings_ms": _project_file_fallback_timings(
                pre_fallback_duration_ms=duration_ms,
                fallback_duration_ms=fallback_duration_ms,
                skip_reason=skip_reason,
                extra_timings=pre_fallback_stage_timings,
            )
        },
    }
    if format == "briefing":
        payload["briefing_degraded"] = True
        payload["briefing_degraded_reason"] = "project_file_fast_path"
    return payload


def project_file_fallback_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    max_packets: int,
    reason: str,
    max_candidates: int = 120,
    topic_scan_chars: int | None = None,
    candidate_read_limit: int | None = None,
    cache: bool = True,
) -> list[dict[str, Any]]:
    scan_chars = _PROJECT_FILE_TOPIC_SCAN_CHARS if topic_scan_chars is None else topic_scan_chars
    packets = _project_file_fallback_packets(
        project_path=project_path,
        topic_hint=topic_hint,
        max_packets=max_packets,
        reason=reason,
        max_candidates=max_candidates,
        topic_scan_chars=scan_chars,
        candidate_read_limit=candidate_read_limit,
    )
    if not packets:
        return []
    if cache:
        cache_project_file_fallback_packet_payloads(
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            packets=packets,
        )
    return packets


def _start_project_file_context_fallback_task(
    *,
    topic_hint: str | None,
    project_path: str | None,
    max_packets: int,
    reason: str,
) -> asyncio.Task[tuple[list[dict[str, Any]], float]] | None:
    if not project_path or max_packets <= 0:
        return None
    task = asyncio.create_task(
        _run_project_file_executor(
            _build_project_file_context_fallback_packets,
            topic_hint=topic_hint,
            project_path=project_path,
            max_packets=max_packets,
            reason=reason,
        )
    )
    task.add_done_callback(_consume_project_file_context_fallback_task)
    return task


def _build_project_file_context_fallback_packets(
    *,
    topic_hint: str | None,
    project_path: str,
    max_packets: int,
    reason: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    packets = project_file_fallback_packet_payloads(
        None,
        group_id="",
        topic_hint=topic_hint,
        project_path=project_path,
        max_packets=max_packets,
        reason=reason,
        cache=False,
    )
    return packets, _elapsed_ms(started)


def _consume_project_file_context_fallback_task(
    task: asyncio.Task[tuple[list[dict[str, Any]], float]],
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        return


def cache_project_file_fallback_packet_payloads(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    packets: Sequence[Mapping[str, Any]],
) -> None:
    """Cache already-built project-file fallback packets without rebuilding them."""
    _cache_project_file_fallback_packets(
        manager,
        group_id=group_id,
        topic_hint=topic_hint,
        project_path=project_path,
        packets=packets,
    )


def _cache_project_file_fallback_packets(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    packets: Sequence[Mapping[str, Any]],
) -> None:
    cache_packets = getattr(manager, "cache_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if (
        not getattr(cfg, "recall_packet_cache_enabled", False)
        or not callable(cache_packets)
        or not packets
    ):
        return
    packets_to_cache = [
        {
            **{key: value for key, value in dict(packet).items() if key != "_cache_scope"},
            "_project_file_fallback_topic_hint": topic_hint,
            "_project_file_fallback_project_path": project_path,
        }
        for packet in packets
    ]
    stable_topic = _derive_context_topic_hint(None, project_path)
    cache_keys = [(topic_hint, project_path)]
    cache_keys.append((stable_topic, project_path))
    seen: set[tuple[str | None, str | None]] = set()
    for cache_topic, cache_project_path in cache_keys:
        if not (cache_topic or cache_project_path):
            continue
        key = (cache_topic, cache_project_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = cache_packets(
                group_id,
                scope="project_home",
                topic_hint=cache_topic,
                project_path=cache_project_path,
                packets=packets_to_cache,
                persist=True,
            )
        except Exception:
            logger.debug("Project file fallback cache write failed", exc_info=True)
            continue
        if inspect.isawaitable(result):
            _close_awaitable(result)


def _cache_loaded_store_context_packets(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    packets: Sequence[Mapping[str, Any]],
) -> None:
    cache_packets = getattr(manager, "cache_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if (
        not getattr(cfg, "recall_packet_cache_enabled", False)
        or not callable(cache_packets)
        or not packets
    ):
        return
    packets_to_cache = [
        {key: value for key, value in dict(packet).items() if key != "_cache_scope"}
        for packet in packets
    ]
    cache_keys = [(topic_hint, project_path), (topic_hint, None)]
    seen: set[tuple[str | None, str | None]] = set()
    for cache_topic, cache_project_path in cache_keys:
        if not (cache_topic or cache_project_path):
            continue
        key = (cache_topic, cache_project_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = cache_packets(
                group_id,
                scope="project_home",
                topic_hint=cache_topic,
                project_path=cache_project_path,
                packets=packets_to_cache,
                persist=True,
            )
        except Exception:
            logger.debug("Loaded-store context packet cache write failed", exc_info=True)
            continue
        if inspect.isawaitable(result):
            _close_awaitable(result)


def _load_cached_context_packets_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
) -> list[dict[str, Any]]:
    get_cached_packets = getattr(manager, "get_cached_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if not getattr(cfg, "recall_packet_cache_enabled", False) or not callable(get_cached_packets):
        return []
    packets: list[dict[str, Any]] = []
    seen_packets: set[tuple[str, str, str]] = set()
    for scope, scope_topic, scope_project in _context_cache_lookup_keys(
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        try:
            hit = get_cached_packets(
                group_id,
                scope=scope,
                topic_hint=scope_topic,
                project_path=scope_project,
                sync_persistent=False,
            )
        except Exception:
            logger.debug("Context surface packet-cache lookup failed", exc_info=True)
            continue
        if inspect.isawaitable(hit):
            _close_awaitable(hit)
            continue
        hit_packets = getattr(hit, "packets", None)
        if not isinstance(hit_packets, list):
            continue
        for packet in hit_packets:
            if isinstance(packet, Mapping):
                _append_unique_cached_packet(
                    packets,
                    seen_packets,
                    _redact_packet_payload(
                        {
                            **dict(packet),
                            "_cache_scope": scope,
                            "_cache_topic_hint": scope_topic,
                            "_cache_project_path": scope_project,
                        }
                    ),
                )
    if _should_attempt_recent_project_file_cache_reuse(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        for packet in _recent_project_file_cache_reuse_packets_from_manager(
            manager,
            group_id=group_id,
            topic_hint=topic_hint,
            project_path=project_path,
            max_packets=_PROJECT_FILE_RECENT_CACHE_REUSE_LIMIT,
        ):
            _append_or_upgrade_cached_packet(packets, seen_packets, packet)
    return packets


def _should_attempt_recent_project_file_cache_reuse(
    packets: Sequence[Mapping[str, Any]],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    if not project_path:
        return False
    if not _topic_specific_context_requested(
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        return False
    relevant_packets = _select_relevant_context_packets(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    )
    for packet in relevant_packets:
        if not isinstance(packet, Mapping):
            continue
        if _context_packet_has_loaded_store_provenance(packet):
            return False
        if _context_packet_is_exact_project_file_cache_hit(
            packet,
            topic_hint=topic_hint,
            project_path=project_path,
        ):
            return False
    return True


def _recent_project_file_cache_reuse_packets_from_manager(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    project_path: str | None,
    max_packets: int,
) -> list[dict[str, Any]]:
    if not project_path or max_packets <= 0:
        return []
    get_recent_packets = getattr(manager, "get_recent_cached_memory_packets", None)
    cfg = _manager_activation_config(manager)
    if not getattr(cfg, "recall_packet_cache_enabled", False) or not callable(get_recent_packets):
        return []
    try:
        recent_packets = get_recent_packets(
            group_id,
            scopes=("project_home",),
            limit_packets=max_packets,
            sync_persistent=True,
        )
    except Exception:
        logger.debug("Recent project-file cache reuse lookup failed", exc_info=True)
        return []
    if inspect.isawaitable(recent_packets):
        _close_awaitable(recent_packets)
        return []
    if not isinstance(recent_packets, list):
        return []
    packets: list[dict[str, Any]] = []
    seen_packets: set[tuple[str, str, str]] = set()
    for packet in recent_packets:
        if len(packets) >= max_packets:
            break
        if not isinstance(packet, Mapping):
            continue
        if not _context_packet_is_project_file_cache_rescue_candidate(
            packet,
            project_path=project_path,
            cache_project_path=project_path,
        ):
            continue
        reuse_packet = _redact_packet_payload(
            {
                **dict(packet),
                "_cache_scope": "project_file_recent_reuse",
                "_project_file_fallback_recent_cache_reuse": True,
            }
        )
        if not _recent_project_file_cache_reuse_packet_relevant(
            reuse_packet,
            topic_hint=topic_hint,
            project_path=project_path,
        ):
            continue
        _append_unique_cached_packet(packets, seen_packets, reuse_packet)
    return _select_relevant_context_packets(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    )


def _recent_project_file_cache_reuse_packet_relevant(
    packet: Mapping[str, Any],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    tokens = _context_relevance_tokens(topic_hint, project_path)
    if not tokens:
        return True
    matches = _context_packet_query_matches(packet, tokens)
    if not matches:
        return False
    required_tokens = _context_required_recent_reuse_tokens(tokens)
    if required_tokens and not required_tokens.issubset(matches):
        return False
    return _context_packet_has_relevance_match(packet, tokens)


def _context_required_recent_reuse_tokens(tokens: set[str]) -> set[str]:
    """Return distinctive query terms that recent project-file reuse must cover."""
    ignored = {
        "cache",
        "cached",
        "codex",
        "context",
        "current",
        "dogfood",
        "final",
        "first",
        "live",
        "matrix",
        "nearby",
        "packet",
        "packets",
        "probe",
        "project",
        "recall",
        "reuse",
        "runtime",
        "startup",
        "trace",
    }
    return {
        token
        for token in tokens
        if not token.isdigit()
        and token not in ignored
        and (len(token) >= 6 or "-" in token or "_" in token)
    }


_PROJECT_FILE_FALLBACK_PATTERNS: tuple[tuple[str, int], ...] = (
    ("README.md", 2400),
    ("MEMORY.md", 3200),
    ("package.json", 2400),
    ("pyproject.toml", 2400),
    ("Makefile", 2400),
    (".env.example", 1800),
    ("docker-compose.yml", 2400),
    ("CLAUDE.md", 2400),
    ("docs/**/*.md", 3200),
    ("notes/**/*.md", 3200),
    ("memory/**/*.md", 3200),
    ("memories/**/*.md", 3200),
    ("skills/**/SKILL.md", 2800),
)
_PROJECT_FILE_TOPIC_SCAN_CHARS = 16_000
_PROJECT_FILE_TOPIC_CANDIDATE_READ_LIMIT = 12
_PROJECT_FILE_RECENT_CACHE_REUSE_LIMIT = 12
_PROJECT_FILE_FALLBACK_PACKET_VERSION = 2
_PROJECT_FILE_PREFIX_CACHE_MAX_ENTRIES = 256
_PROJECT_FILE_PREFIX_CACHE: dict[str, tuple[int, int, int, str]] = {}
_PROJECT_FILE_PREFIX_WARMUP_TASKS: dict[str, asyncio.Task[None]] = {}
_PROJECT_FILE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="engram-project-files",
)
_PROJECT_FILE_FALLBACK_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


async def _run_project_file_executor(
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _PROJECT_FILE_EXECUTOR,
        partial(function, *args, **kwargs),
    )


def schedule_project_file_prefix_warmup(
    project_path: str | None,
    *,
    topic_hint: str | None = None,
) -> bool:
    """Warm local project-file prefixes without touching graph or packet cache."""
    if not project_path:
        return False
    try:
        project_dir = Path(project_path).expanduser()
        resolved_project_dir = project_dir.resolve()
        home_dir = Path.home().resolve()
        if (
            not resolved_project_dir.is_dir()
            or resolved_project_dir == home_dir
            or str(resolved_project_dir) == "/"
        ):
            return False
    except OSError:
        return False

    warm_topic = (topic_hint or resolved_project_dir.name).strip()
    if not warm_topic:
        warm_topic = resolved_project_dir.name
    task_key = f"{resolved_project_dir}::{warm_topic}"
    existing_task = _PROJECT_FILE_PREFIX_WARMUP_TASKS.get(task_key)
    if existing_task is not None and not existing_task.done():
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    task = loop.create_task(
        _run_project_file_executor(
            _warm_project_file_prefix_cache,
            str(resolved_project_dir),
            topic_hint=warm_topic,
        )
    )
    _PROJECT_FILE_PREFIX_WARMUP_TASKS[task_key] = task
    task.add_done_callback(
        lambda done_task, key=task_key: _consume_project_file_prefix_warmup_task(key, done_task)
    )
    return True


def _warm_project_file_prefix_cache(
    project_path: str,
    *,
    topic_hint: str | None = None,
) -> None:
    try:
        project_dir = Path(project_path).expanduser()
        if not project_dir.is_dir() or str(project_dir) in {str(Path.home()), "/"}:
            return
    except OSError:
        return
    warm_topic = (topic_hint or project_dir.name).strip()
    candidate_paths = _iter_project_file_fallback_candidates(project_dir)
    candidate_paths = _rank_project_file_fallback_candidates(
        candidate_paths,
        project_name=project_dir.name,
        topic_hint=warm_topic,
        max_reads=_PROJECT_FILE_TOPIC_CANDIDATE_READ_LIMIT,
    )
    for path, _rel_path, max_chars in candidate_paths:
        read_limit = max(max_chars, _PROJECT_FILE_TOPIC_SCAN_CHARS)
        try:
            _read_project_file_prefix(path, read_limit)
        except OSError:
            continue


def _consume_project_file_prefix_warmup_task(
    task_key: str,
    task: asyncio.Task[None],
) -> None:
    if _PROJECT_FILE_PREFIX_WARMUP_TASKS.get(task_key) is task:
        _PROJECT_FILE_PREFIX_WARMUP_TASKS.pop(task_key, None)
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("Project file prefix warmup failed", exc_info=True)


def _project_file_fallback_packets(
    *,
    project_path: str | None,
    topic_hint: str | None,
    max_packets: int,
    reason: str,
    max_candidates: int = 120,
    topic_scan_chars: int = _PROJECT_FILE_TOPIC_SCAN_CHARS,
    candidate_read_limit: int | None = None,
) -> list[dict[str, Any]]:
    if not project_path or max_packets <= 0:
        return []
    project_dir = Path(project_path).expanduser()
    try:
        if not project_dir.is_dir() or str(project_dir) in {str(Path.home()), "/"}:
            return []
    except OSError:
        return []

    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    candidate_paths = _iter_project_file_fallback_candidates(
        project_dir,
        max_candidates=max_candidates,
    )
    candidate_paths = _rank_project_file_fallback_candidates(
        candidate_paths,
        project_name=project_dir.name,
        topic_hint=topic_hint,
        max_reads=min(
            max_candidates,
            candidate_read_limit or _PROJECT_FILE_TOPIC_CANDIDATE_READ_LIMIT,
        )
        if topic_hint
        else max_candidates,
    )
    for path, rel_path, max_chars in candidate_paths:
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        read_limit = max(max_chars, topic_scan_chars) if topic_hint else max_chars
        try:
            raw_content = _read_project_file_prefix(path, read_limit)
        except OSError:
            continue
        content = raw_content[:max_chars]
        search_content = raw_content if topic_hint else content
        matched_lines = (
            _project_file_matching_lines(
                search_content,
                topic_hint=topic_hint,
                limit=3,
            )
            if topic_hint
            else []
        )
        summary = _project_file_summary(
            project_dir.name,
            rel_path,
            search_content,
            topic_hint=topic_hint,
            matched_lines=matched_lines[:1],
        )
        claims = _project_file_claims(
            search_content,
            topic_hint=topic_hint,
            matched_lines=matched_lines,
        )
        packet = {
            "packet_type": "project_home",
            "title": _redact_packet_text(f"Project File: {rel_path}"),
            "summary": _redact_packet_text(summary),
            "why_now": reason,
            "confidence": 0.65,
            "entity_ids": [],
            "relationship_ids": [],
            "episode_ids": [],
            "evidence_lines": [_redact_packet_text(claim) for claim in claims],
            "provenance": [f"file:{rel_path}"],
            "supporting_intents": ["project_file_context_fallback"],
            "trust": {
                "freshness": "local",
                "source": "project_file",
                "confidence": 0.65,
                "why_now": reason,
                "provenance_count": 1,
                "evidence_count": len(claims),
                "belief_status": "unknown",
                "confirmed_count": 0,
                "corrected_count": 0,
                "dismissed_count": 0,
                "last_confirmed_at": None,
                "last_corrected_at": None,
                "last_dismissed_at": None,
            },
            "_cache_scope": "project_file_fallback",
            "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
        }
        score = _project_file_score(
            topic_hint=topic_hint,
            project_name=project_dir.name,
            rel_path=rel_path,
            content=search_content,
        )
        priority = _artifact_context_priority(rel_path)
        candidates.append((score, priority, rel_path, packet))

    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = candidates[:max_packets]
    if all(score <= 0 for score, _priority, _rel_path, _packet in selected):
        selected.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [_redact_packet_payload(packet) for _score, _priority, _rel_path, packet in selected]


def _read_project_file_prefix(path: Path, char_limit: int) -> str:
    normalized_limit = max(1, int(char_limit))
    stat = path.stat()
    cache_key = str(path)
    cached = _PROJECT_FILE_PREFIX_CACHE.get(cache_key)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_limit, cached_content = cached
        if (
            cached_mtime_ns == stat.st_mtime_ns
            and cached_size == stat.st_size
            and cached_limit >= normalized_limit
        ):
            return cached_content[:normalized_limit]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(normalized_limit)
    _PROJECT_FILE_PREFIX_CACHE[cache_key] = (
        stat.st_mtime_ns,
        stat.st_size,
        normalized_limit,
        content,
    )
    _trim_project_file_prefix_cache()
    return content


def _trim_project_file_prefix_cache() -> None:
    overflow = len(_PROJECT_FILE_PREFIX_CACHE) - _PROJECT_FILE_PREFIX_CACHE_MAX_ENTRIES
    if overflow <= 0:
        return
    for key in list(_PROJECT_FILE_PREFIX_CACHE)[:overflow]:
        _PROJECT_FILE_PREFIX_CACHE.pop(key, None)


def _project_file_fallback_reason(skip_reason: str | None) -> str:
    if skip_reason == "cache_relevance_miss":
        return (
            "Cached project packets did not match the specific topic, so this "
            "packet was synthesized from local project files without loaded-store reads."
        )
    if skip_reason == "context_timeout":
        return (
            "Graph context timed out; this packet was synthesized from local "
            "project files without loaded-store reads."
        )
    return "This packet was synthesized from local project files without loaded-store reads."


def _project_file_fallback_timings(
    *,
    pre_fallback_duration_ms: float,
    fallback_duration_ms: float,
    skip_reason: str | None,
    extra_timings: Mapping[str, float | None] | None = None,
) -> dict[str, float]:
    timings = {"project_file_fallback": fallback_duration_ms}
    if extra_timings:
        timings.update(
            {key: value for key, value in extra_timings.items() if isinstance(value, int | float)}
        )
    if skip_reason:
        timings[skip_reason] = pre_fallback_duration_ms
    return timings


def _iter_project_file_fallback_candidates(
    project_dir: Path,
    *,
    max_candidates: int = 120,
) -> list[tuple[Path, str, int]]:
    candidates: list[tuple[Path, str, int]] = []
    for pattern, max_chars in _PROJECT_FILE_FALLBACK_PATTERNS:
        for path in sorted(project_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                rel_path = path.relative_to(project_dir).as_posix()
            except ValueError:
                continue
            if any(part in _PROJECT_FILE_FALLBACK_EXCLUDED_PARTS for part in path.parts):
                continue
            candidates.append((path, rel_path, max_chars))
            if len(candidates) >= max(1, max_candidates):
                return candidates
    return candidates


def _rank_project_file_fallback_candidates(
    candidates: Sequence[tuple[Path, str, int]],
    *,
    project_name: str,
    topic_hint: str | None,
    max_reads: int,
) -> list[tuple[Path, str, int]]:
    if not topic_hint or max_reads <= 0:
        return list(candidates)
    terms = _project_file_relevance_terms(topic_hint, project_name=project_name)
    if not terms:
        return list(candidates[:max_reads])
    ranked: list[tuple[int, int, int, tuple[Path, str, int]]] = []
    for index, candidate in enumerate(candidates):
        _path, rel_path, _max_chars = candidate
        rel_text = rel_path.lower()
        path_score = sum(8 for term in terms if term in rel_text)
        ranked.append(
            (
                path_score,
                _artifact_context_priority(rel_path),
                -index,
                candidate,
            )
        )
    ranked.sort(reverse=True)
    return [candidate for _score, _priority, _index, candidate in ranked[:max_reads]]


def _project_file_score(
    *,
    topic_hint: str | None,
    project_name: str,
    rel_path: str,
    content: str,
) -> float:
    terms = _project_file_relevance_terms(topic_hint, project_name=project_name)
    if not terms:
        return 0.0
    rel_text = rel_path.lower()
    body_text = content.lower()
    path_score = sum(1 for term in terms if term in rel_text)
    body_score = sum(1 for term in terms if term in body_text)
    relevance_score = path_score + body_score
    if relevance_score <= 0:
        return 0.0
    priority_boost = _artifact_context_priority(rel_path) / 20.0
    return float(relevance_score + priority_boost)


def _topic_specific_context_requested(
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    return bool(_context_relevance_tokens(topic_hint, project_path))


def _fast_project_context_requested(
    *,
    operation_source: str,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    return operation_source == "axi_context" and bool(project_path)


def _context_cached_packets_relevant(
    packets: Sequence[Mapping[str, Any]],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    tokens = _context_relevance_tokens(topic_hint, project_path)
    if not tokens:
        return True
    return any(_context_packet_has_relevance_match(packet, tokens) for packet in packets)


def _select_relevant_context_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> list[dict[str, Any]]:
    """Return only packets that should render for a topic-specific context query."""
    packet_list = [dict(packet) for packet in packets]
    tokens = _context_relevance_tokens(topic_hint, project_path)
    if not tokens:
        return packet_list
    specific_tokens = _context_specific_relevance_tokens(tokens)
    exact_project_file_packets = [
        packet
        for packet in packet_list
        if _context_packet_is_exact_project_file_cache_hit(
            packet,
            topic_hint=topic_hint,
            project_path=project_path,
        )
    ]
    specifically_relevant = [
        packet
        for packet in packet_list
        if packet in exact_project_file_packets
        or (
            specific_tokens
            and _context_packet_has_relevance_match(packet, tokens)
            and _context_packet_query_matches(packet, specific_tokens)
        )
    ]
    if specifically_relevant:
        packet_list = specifically_relevant
    relevant = [
        packet
        for packet in packet_list
        if packet in exact_project_file_packets
        or _context_packet_has_relevance_match(packet, tokens)
    ]
    if not relevant:
        return []
    # Durable graph facts always win over session recap packets.
    durable_relevant = [packet for packet in relevant if _context_packet_is_durable_fact(packet)]
    if durable_relevant:
        # Keep durable first, then other non-recap, then recap as filler.
        non_recap = [
            packet
            for packet in relevant
            if not _context_packet_is_session_recent(packet) and packet not in durable_relevant
        ]
        return durable_relevant + non_recap

    if _session_recent_packets_strongly_answer_topic(
        relevant,
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        # Only short-circuit to session_recent when NO durable facts matched.
        session_recent_relevant = [
            packet
            for packet in relevant
            if _context_packet_is_session_recent(packet)
            and _context_packet_has_relevance_match(packet, tokens)
        ]
        if session_recent_relevant and not any(
            _context_packet_is_durable_fact(packet) for packet in relevant
        ):
            return session_recent_relevant
    return relevant


def _can_enrich_session_recent_context(
    manager: Any,
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    return _topic_specific_context_requested(
        topic_hint=topic_hint,
        project_path=project_path,
    ) and callable(getattr(manager, "fast_recall_fallback", None))


def _context_packets_need_loaded_store_enrichment(
    packets: Sequence[Mapping[str, Any]],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    if _session_recent_packets_strongly_answer_topic(
        packets,
        topic_hint=topic_hint,
        project_path=project_path,
    ):
        return False
    saw_packet = False
    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        saw_packet = True
        if _context_packet_has_loaded_store_provenance(packet):
            return False
        if _context_packet_is_exact_project_file_cache_hit(
            packet,
            topic_hint=topic_hint,
            project_path=project_path,
        ):
            return False
        if _context_packet_is_recent_project_file_cache_reuse_hit(
            packet,
            project_path=project_path,
        ):
            return False
    return saw_packet


def _session_recent_packets_strongly_answer_topic(
    packets: Sequence[Mapping[str, Any]],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    tokens = _context_relevance_tokens(topic_hint, project_path)
    if not tokens:
        return False
    anchor_tokens = _context_session_recent_anchor_tokens(tokens)
    if not anchor_tokens:
        return False

    matches: set[str] = set()
    for packet in packets:
        if not isinstance(packet, Mapping) or not _context_packet_is_session_recent(packet):
            continue
        matches.update(_context_packet_query_matches(packet, tokens))

    anchor_matches = matches & anchor_tokens
    if not anchor_matches:
        return False
    if any(not token.isdigit() for token in anchor_matches):
        return len(matches) >= (1 if len(tokens) <= 2 else 2)
    return len(anchor_matches) >= 2 or len(matches) >= 3


def _context_packet_is_session_recent(packet: Mapping[str, Any]) -> bool:
    if packet.get("_cache_scope") == SESSION_RECENT_PACKET_SCOPE:
        return True
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    return packet_type == "recent_observation"


def _context_packet_is_durable_fact(packet: Mapping[str, Any]) -> bool:
    """True for graph-backed durable packets (not session recap or latent cues)."""
    from engram.extraction.promotion import is_durable_recall_entity_type

    if _context_packet_is_session_recent(packet):
        return False
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    if packet_type in {
        "cue_packet",
        "recent_observation",
        "episode_packet",
        "recall_diagnostic",
    }:
        return False
    # Entity-derived packets: fact/state/open_loop/intention with entity ids.
    entity_ids = packet.get("entity_ids") or packet.get("entityIds") or []
    if entity_ids and packet_type in {
        "fact_packet",
        "state_packet",
        "open_loop_packet",
        "intention_packet",
        "identity_core",
        "identityCore",
    }:
        return True
    # Title/summary heuristics for Decision/Preference style facts.
    title = str(packet.get("title") or "")
    summary = str(packet.get("summary") or "")
    blob = f"{title} {summary}".lower()
    if any(
        token in blob
        for token in (
            "decision:",
            "preference:",
            "correction:",
            "person:",
            "fact:",
            "goal:",
        )
    ):
        return True
    trust = packet.get("trust")
    if isinstance(trust, Mapping):
        source = str(trust.get("source") or "")
        if source == "entity":
            return True
        if source in {"api_auto_observe", "mcp_observe", "cue"}:
            return False
    # Provenance entity: markers.
    provenance = packet.get("provenance") or []
    if any(str(p).startswith("entity:") for p in provenance):
        return True
    # entity_type if present on packet
    entity_type = str(packet.get("entity_type") or packet.get("entityType") or "")
    return is_durable_recall_entity_type(entity_type)


def _context_session_recent_anchor_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if any(char.isdigit() for char in token)
        or "_" in token
        or (len(token) >= 12 and token.isalnum())
    }


def _context_packet_is_exact_project_file_cache_hit(
    packet: Mapping[str, Any],
    *,
    topic_hint: str | None,
    project_path: str | None,
) -> bool:
    """Allow topic-specific project-file packets to satisfy repeat context calls."""
    trust = packet.get("trust")
    source = ""
    if isinstance(trust, Mapping):
        source = str(trust.get("source") or "")
    if source != "project_file":
        return False
    if packet.get("_project_file_fallback_version") != _PROJECT_FILE_FALLBACK_PACKET_VERSION:
        return False
    return _normalize_context_cache_value(
        packet.get("_project_file_fallback_topic_hint")
    ) == _normalize_context_cache_value(topic_hint) and _normalize_context_cache_value(
        packet.get("_project_file_fallback_project_path")
    ) == _normalize_context_cache_value(project_path)


def _context_packet_is_recent_project_file_cache_reuse_hit(
    packet: Mapping[str, Any],
    *,
    project_path: str | None,
) -> bool:
    if packet.get("_cache_scope") != "project_file_recent_reuse":
        return False
    if packet.get("_project_file_fallback_recent_cache_reuse") is not True:
        return False
    return _context_packet_is_project_file_cache_rescue_candidate(
        packet,
        project_path=project_path,
        cache_project_path=project_path,
    )


def _context_packet_is_project_file_cache_rescue_candidate(
    packet: Mapping[str, Any],
    *,
    project_path: str | None,
    cache_project_path: str | None,
) -> bool:
    """Allow only current project-file packets from the same project as fallback rescue."""
    trust = packet.get("trust")
    source = ""
    if isinstance(trust, Mapping):
        source = str(trust.get("source") or "")
    if source != "project_file":
        return False
    if packet.get("_project_file_fallback_version") != _PROJECT_FILE_FALLBACK_PACKET_VERSION:
        return False
    expected_path = _normalize_context_cache_value(project_path)
    packet_path = _normalize_context_cache_value(packet.get("_project_file_fallback_project_path"))
    cache_path = _normalize_context_cache_value(cache_project_path)
    if not expected_path:
        return False
    if packet_path and packet_path != expected_path:
        return False
    return cache_path == expected_path or packet_path == expected_path


def _normalize_context_cache_value(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _context_packet_has_loaded_store_provenance(packet: Mapping[str, Any]) -> bool:
    trust = packet.get("trust")
    trust_source = ""
    if isinstance(trust, Mapping):
        trust_source = str(trust.get("source") or trust.get("sourceType") or "")
    if trust_source in {"project_file", "mcp_observe", "api_auto_observe"}:
        return False
    if trust_source:
        return True
    provenance = packet.get("provenance")
    if isinstance(provenance, list | tuple):
        return any(
            str(item).startswith(("cue:", "episode:", "entity:", "relationship:"))
            for item in provenance
        )
    return False


def _prefer_project_context_results(
    results: Sequence[Mapping[str, Any]],
    *,
    project_path: str | None,
) -> list[dict[str, Any]]:
    result_list = [dict(result) for result in results]
    if not project_path:
        return result_list
    project_name = Path(project_path).expanduser().name.strip().lower()
    if not project_name:
        return result_list
    project_path_text = str(Path(project_path).expanduser()).lower()
    project_results = [
        result
        for result in result_list
        if _context_result_mentions_project(
            result,
            project_name=project_name,
            project_path=project_path_text,
        )
    ]
    return project_results or result_list


def _context_result_mentions_project(
    result: Mapping[str, Any],
    *,
    project_name: str,
    project_path: str,
) -> bool:
    text_parts: list[str] = []
    episode = result.get("episode")
    if isinstance(episode, Mapping):
        text_parts.extend(str(episode.get(key) or "") for key in ("content", "source", "id"))
    cue = result.get("cue")
    if isinstance(cue, Mapping):
        text_parts.append(str(cue.get("cue_text") or ""))
        spans = cue.get("supporting_spans")
        if isinstance(spans, list | tuple):
            text_parts.extend(str(span) for span in spans)
    for linked in result.get("linked_entities") or []:
        if isinstance(linked, Mapping):
            text_parts.extend(str(linked.get(key) or "") for key in ("name", "summary", "id"))
        else:
            text_parts.append(str(linked))
    text = " ".join(text_parts).lower()
    return project_name in text or bool(project_path and project_path in text)


def _context_specific_relevance_tokens(tokens: set[str]) -> set[str]:
    """Return high-signal query tokens that should dominate broad topic matches."""
    return {
        token
        for token in tokens
        if any(char.isdigit() for char in token) or "-" in token or "_" in token or len(token) >= 10
    }


def _context_packet_query_score(packet: Mapping[str, Any], tokens: set[str]) -> int:
    return len(_context_packet_query_matches(packet, tokens))


def _context_packet_query_matches(
    packet: Mapping[str, Any],
    tokens: set[str],
) -> set[str]:
    text_parts: list[str] = []
    for key in (
        "title",
        "summary",
        "packet_type",
        "packetType",
        "provenance",
        "supporting_intents",
        "supportingIntents",
        "evidence_lines",
        "evidenceLines",
    ):
        value = packet.get(key)
        if isinstance(value, list | tuple):
            text_parts.extend(str(item) for item in value)
        else:
            text_parts.append(str(value or ""))
    text = " ".join(text_parts).lower()
    return {token for token in tokens if token in text}


def _context_packet_has_relevance_match(
    packet: Mapping[str, Any],
    tokens: set[str],
) -> bool:
    """Require more than a lone date/id match for topic-specific cache reuse."""
    matches = _context_packet_query_matches(packet, tokens)
    if not matches:
        return False
    specific_tokens = _context_specific_relevance_tokens(tokens)
    if not specific_tokens:
        return True
    specific_matches = matches & specific_tokens
    if not specific_matches:
        return len(matches) >= 2
    if any(not token.isdigit() for token in specific_matches):
        return True
    return len(matches) >= 2


def _context_relevance_tokens(
    topic_hint: str | None,
    project_path: str | None,
) -> set[str]:
    project_terms: set[str] = set()
    if project_path:
        project_name = Path(project_path).expanduser().name.lower()
        if project_name:
            project_terms.add(project_name)
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (topic_hint or "").lower())
        if term
        not in {
            "and",
            "for",
            "from",
            "the",
            "this",
            "that",
            "with",
            "project",
            "context",
            "goal",
            "continuation",
        }
        and term not in project_terms
    }


def _project_file_relevance_terms(
    topic_hint: str | None,
    *,
    project_name: str | None = None,
) -> set[str]:
    ignored = {
        "and",
        "for",
        "from",
        "the",
        "this",
        "that",
        "with",
        "project",
        "startup",
        "context",
        "goal",
        "continuation",
    }
    if project_name:
        ignored.add(project_name.lower())
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (topic_hint or "").lower())
        if term and term not in ignored
    }


def _project_file_matching_lines(
    content: str,
    *,
    topic_hint: str | None,
    limit: int,
) -> list[str]:
    terms = _project_file_relevance_terms(topic_hint)
    if not terms or limit <= 0:
        return []
    lines = content.splitlines()
    matches: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        window = _project_file_matching_window(lines, index)
        text = window.lower()
        score = sum(1 for term in terms if term in text)
        if score <= 0:
            continue
        current_score = sum(1 for term in terms if term in stripped.lower())
        if current_score <= 0:
            continue
        claim_text = window if window != stripped else stripped
        claim = _project_file_claim_from_line(claim_text)
        if claim in seen:
            continue
        seen.add(claim)
        matches.append((score, index, claim))
    matches.sort(reverse=True)
    return [claim for _score, _index, claim in matches[:limit]]


def _project_file_matching_window(lines: Sequence[str], index: int) -> str:
    current = lines[index].strip()
    if not current:
        return current
    if current.startswith("#"):
        return current
    next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
    parts = [current]
    previous_index = index - 1
    continuation_line = current
    while previous_index >= 0 and len(parts) < 4:
        previous_part = lines[previous_index].strip()
        if (
            not previous_part
            or previous_part.startswith("#")
            or not _project_file_line_starts_continuation(
                continuation_line,
                previous_line=previous_part,
            )
        ):
            break
        fragment = _project_file_previous_continuation_fragment(previous_part)
        parts.insert(0, fragment)
        if fragment != previous_part:
            break
        continuation_line = previous_part
        previous_index -= 1
    if next_line and _project_file_line_invites_continuation(
        current,
        next_line=next_line,
    ):
        parts.append(next_line)
    parts = [part for part in parts if part and not part.startswith("#")]
    if not parts:
        return current
    return _truncate_packet_text(" ".join(parts), 240)


def _project_file_line_starts_continuation(line: str, *, previous_line: str = "") -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped.startswith(("`", ")", "]", "}", ",", ".", ";", ":")):
        return True
    if (
        previous_line
        and stripped[:1].islower()
        and not previous_line.rstrip().endswith((".", "!", "?", ";", ":"))
    ):
        return True
    first = stripped.split(maxsplit=1)[0].lower().strip(",.;:")
    return first in {"and", "or", "with", "in", "at", "for", "to", "from"}


def _project_file_previous_continuation_fragment(line: str) -> str:
    stripped = line.strip()
    boundaries = [stripped.rfind(marker) for marker in (". ", "! ", "? ")]
    boundary = max(boundaries)
    if boundary <= 0:
        return stripped
    fragment = stripped[boundary + 2 :].strip()
    if len(fragment) >= 20:
        return fragment
    return stripped


def _project_file_line_invites_continuation(line: str, *, next_line: str) -> bool:
    stripped = line.rstrip()
    following = next_line.lstrip()
    if not stripped or not following:
        return False
    if following.startswith(("`", "(", "[", "{", ",", ".", ";", ":")):
        return True
    last = stripped.split()[-1].lower().strip("`'\"()[]{}.,;:")
    return last in {
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "produced",
        "to",
        "via",
        "with",
    }


def _project_file_summary(
    project_name: str,
    rel_path: str,
    content: str,
    *,
    topic_hint: str | None = None,
    matched_lines: Sequence[str] | None = None,
) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    heading_lines = [
        line.lstrip("#").strip()
        for line in lines
        if line.startswith("#") and line.lstrip("#").strip()
    ]
    if matched_lines is None:
        matched_lines = _project_file_matching_lines(
            content,
            topic_hint=topic_hint,
            limit=1,
        )
    if matched_lines:
        if heading_lines:
            return (
                f"{project_name} file {rel_path} — {heading_lines[0]}; "
                f"{_truncate_packet_text(matched_lines[0], 200)}"
            )
        return f"{project_name} file {rel_path} — {_truncate_packet_text(matched_lines[0], 240)}"
    if heading_lines:
        return f"{project_name} file {rel_path} — " + "; ".join(heading_lines[:3])
    if lines:
        return f"{project_name} file {rel_path} — {_truncate_packet_text(lines[0], 240)}"
    return f"{project_name} file {rel_path}"


def _project_file_claim_from_line(stripped: str) -> str:
    if stripped.startswith("#"):
        heading = stripped.lstrip("#").strip()
        return f"heading={heading}" if heading else _truncate_packet_text(stripped, 180)
    return _truncate_packet_text(stripped, 180)


def _project_file_claims(
    content: str,
    *,
    topic_hint: str | None = None,
    matched_lines: Sequence[str] | None = None,
) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()

    def append_claim(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            claims.append(value)

    topic_matches = (
        list(matched_lines)
        if matched_lines is not None
        else _project_file_matching_lines(content, topic_hint=topic_hint, limit=3)
    )
    for claim in topic_matches[:3]:
        append_claim(claim)

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            append_claim(_project_file_claim_from_line(stripped))
        elif stripped.startswith(("-", "*")):
            append_claim(_project_file_claim_from_line(stripped))
        if len(claims) >= 3:
            break
    return claims


def _context_entity_packet(
    entity: Mapping[str, Any],
    *,
    packet_type: str,
    why_now: str,
) -> dict[str, Any]:
    entity_id = str(entity.get("id") or "")
    facts = [_redact_packet_text(str(fact)) for fact in entity.get("facts", [])[:3]]
    summary = _redact_packet_text(str(entity.get("summary") or "").strip())
    title = _redact_packet_text(
        f"{packet_type.replace('_', ' ').title()}: {entity.get('name') or 'Memory'}"
    )
    packet = {
        "packet_type": packet_type,
        "title": title,
        "summary": summary or "; ".join(facts)[:180],
        "why_now": why_now,
        "confidence": 0.8,
        "entity_ids": [entity_id] if entity_id else [],
        "relationship_ids": [],
        "episode_ids": [],
        "evidence_lines": facts,
        "provenance": [f"entity:{entity_id}"] if entity_id else [],
        "supporting_intents": ["context_cache"],
        "trust": {
            "freshness": "recent",
            "source": "cache",
            "confidence": 0.8,
            "why_now": why_now,
            "provenance_count": 1 if entity_id else 0,
            "evidence_count": len(facts),
            "belief_status": "unknown",
            "confirmed_count": 0,
            "corrected_count": 0,
            "dismissed_count": 0,
            "last_confirmed_at": None,
            "last_corrected_at": None,
            "last_dismissed_at": None,
        },
    }
    return packet


def _redact_packet_text(text: str) -> str:
    """Remove obvious credential-shaped substrings from reusable context packets."""
    if not text:
        return text
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}{' ' if match.group(2) == ':' else ''}[redacted]"
        ),
        text,
    )
    redacted = _SECRET_TOKEN_RE.sub("[redacted-token]", redacted)
    return redacted


def _truncate_packet_text(text: str, max_chars: int) -> str:
    """Trim reusable packet text without cutting through the final word."""
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    limit = max_chars - 3
    head = normalized[:limit].rstrip()
    boundary = head.rfind(" ")
    if boundary >= int(limit * 0.6):
        head = head[:boundary].rstrip()
    return head.rstrip(".,;:") + "..."


def _redact_packet_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_packet_text(value)
    if isinstance(value, Mapping):
        return {key: _redact_packet_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_packet_payload(item) for item in value]
    return value


def _append_unique_cached_packet(
    packets: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    packet: Any,
) -> None:
    if not isinstance(packet, dict):
        return
    fingerprint = _packet_fingerprint(packet)
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    packets.append(packet)


def _append_or_upgrade_cached_packet(
    packets: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    packet: Any,
) -> None:
    if not isinstance(packet, dict):
        return
    fingerprint = _packet_fingerprint(packet)
    if fingerprint not in seen:
        seen.add(fingerprint)
        packets.append(packet)
        return
    for existing in packets:
        if _packet_fingerprint(existing) == fingerprint:
            existing.update(packet)
            return


def _packet_fingerprint(packet: Mapping[str, Any]) -> tuple[str, str, str]:
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    title = str(packet.get("title") or "")
    entity_ids = packet.get("entity_ids") or packet.get("entityIds") or []
    if isinstance(entity_ids, list) and entity_ids:
        source = ",".join(str(entity_id) for entity_id in entity_ids)
    else:
        source = str(packet.get("summary") or "")[:160]
    return packet_type, title, source


def _packet_scope_counts(packets: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for packet in packets:
        scope = str(packet.get("_cache_scope") or "unknown")
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _manager_activation_config(manager: Any) -> ActivationConfig:
    get_cfg = getattr(manager, "get_activation_config", None)
    if callable(get_cfg):
        try:
            cfg = get_cfg()
        except Exception:
            cfg = None
        if inspect.isawaitable(cfg):
            _close_awaitable(cfg)
            cfg = None
        if isinstance(cfg, ActivationConfig):
            return cfg
    get_cfg = getattr(manager, "get_memory_need_config", None)
    if callable(get_cfg):
        try:
            cfg = get_cfg()
        except Exception:
            cfg = None
        if inspect.isawaitable(cfg):
            _close_awaitable(cfg)
            cfg = None
        if isinstance(cfg, ActivationConfig):
            return cfg
    return ActivationConfig()


async def _record_context_timeout(
    manager: Any,
    *,
    group_id: str,
    operation_source: str,
    budget: RecallBudget,
    status: str,
    skip_reason: str,
    duration_ms: float,
    timeout: bool,
) -> None:
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="context",
            source=operation_source,
            mode=operation_source,
            status=status,
            duration_ms=duration_ms,
            skip_reason=skip_reason,
            timeout=timeout,
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            budget_miss=True,
            degraded=bool(timeout or budget.timeout_degrades),
        ),
    )


def _context_timeout_payload(
    *,
    format: str,
    budget: RecallBudget,
    duration_ms: float,
    status: str,
    skip_reason: str,
    timeout: bool,
) -> dict[str, Any]:
    context = (
        "## Active Memory Context\n\n"
        "Context lookup degraded before fresh memory context could be assembled."
    )
    payload: dict[str, Any] = {
        "context": context,
        "entity_count": 0,
        "fact_count": 0,
        "token_estimate": MemoryContextBuilder.estimate_tokens(context),
        "format": "structured",
        "status": status,
        "budget": {
            "profile": budget.profile,
            "surface": budget.surface,
            "mode": budget.mode,
            "max_wall_ms": budget.max_wall_ms,
            "duration_ms": duration_ms,
            "budget_miss": True,
            "timeout": timeout,
            "degraded": True,
            "skip_reason": skip_reason,
        },
        "lifecycle": {
            "stage": "recall",
            "degraded": True,
            "timeout": timeout,
            "skip_reason": skip_reason,
        },
        "diagnostics": {
            "stage_timings_ms": {
                "context_timeout": duration_ms,
            }
        },
        "cached_packets": [],
        "packet_cache": {"hit": False, "packet_count": 0, "scopes": {}},
    }
    if format == "briefing":
        payload["briefing_degraded"] = True
        payload["briefing_degraded_reason"] = "context_timeout"
    return payload


def _close_awaitable(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
