"""Shared auto-recall policy helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig
from engram.ingestion.capture_surface import store_observation
from engram.retrieval.budgets import recall_budget_for_profile
from engram.retrieval.context import (
    manager_conversation_context,
    manager_conversation_recent_turns,
    manager_conversation_top_entity_names,
)
from engram.retrieval.context_builder import MemoryContextBuilder
from engram.retrieval.control import (
    record_manager_memory_need_analysis,
    resolve_manager_recall_need_thresholds,
)
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.memory_operations import (
    MemoryOperationSample,
    record_manager_memory_operation,
)
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.recall_surface import (
    EXPLICIT_RECALL_PACKET_CACHE_SCOPE,
    _filter_packets_for_query,
)

logger = logging.getLogger(__name__)

RECALL_TOOLS = frozenset(
    {
        "observe",
        "remember",
        "recall",
        "search_entities",
        "search_facts",
        "get_context",
        "route_question",
        "search_artifacts",
    }
)
WRITE_TOOLS = frozenset({"observe", "remember"})


@dataclass(frozen=True)
class SessionPrimePlan:
    """Context-prime request derived from auto-recall session policy."""

    topic_hint: str | None
    max_tokens: int


@dataclass(frozen=True)
class SessionPrimeSurface:
    """Context-prime result plus session-state mutation intent."""

    context: dict[str, Any] | None
    should_mark_primed: bool


@dataclass(frozen=True)
class RecallMiddlewarePlan:
    """Transport-neutral decisions for MCP recall piggyback middleware."""

    should_recall: bool
    surface_notifications_when_recall_disabled: bool
    auto_observe_content: bool
    ingest_live_turn: bool
    cache_only: bool = False


class RecallCooldown:
    """Rate limiter plus topic deduplication for auto-recall."""

    def __init__(self, max_per_minute: int = 3, cooldown_seconds: float = 60.0) -> None:
        self._entries: deque[tuple[set[str], float]] = deque(maxlen=20)
        self.max_per_minute = max_per_minute
        self.cooldown_seconds = cooldown_seconds

    def _tokenize(self, query: str) -> set[str]:
        return {word.lower() for word in query.split() if len(word) > 2}

    def is_throttled(self, query: str, now: float) -> bool:
        recent = [timestamp for _, timestamp in self._entries if now - timestamp < 60.0]
        if len(recent) >= self.max_per_minute:
            return True

        tokens = self._tokenize(query)
        if not tokens:
            return False
        for previous_tokens, timestamp in self._entries:
            if now - timestamp > self.cooldown_seconds:
                continue
            if not previous_tokens:
                continue
            overlap = len(tokens & previous_tokens) / max(len(tokens | previous_tokens), 1)
            if overlap > 0.5:
                return True
        return False

    def record(self, query: str, now: float) -> None:
        self._entries.append((self._tokenize(query), now))


def extract_recall_query(content: str) -> str:
    """Extract a compact recall query from content."""
    if len(content) < 20:
        return ""

    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
    if proper_nouns:
        return " ".join(proper_nouns)[:200]

    first_sentence = content.split(".")[0].strip()
    return first_sentence[:200]


def should_recall_for_tool(tool_name: str, cfg: ActivationConfig | None) -> bool:
    """Return whether an MCP tool call should receive auto-recall context."""
    if not cfg or tool_name not in RECALL_TOOLS:
        return False
    if tool_name == "observe":
        return bool(cfg.auto_recall_on_observe)
    if tool_name == "remember":
        return bool(cfg.auto_recall_on_remember)
    return bool(cfg.auto_recall_on_tool_call)


def plan_session_prime(
    content: str | None,
    cfg: ActivationConfig,
    *,
    already_primed: bool,
) -> SessionPrimePlan | None:
    """Plan the first-call MCP session prime without reaching into transport state."""
    if not cfg.auto_recall_session_prime or already_primed:
        return None

    topic = extract_recall_query(content) if content else ""
    return SessionPrimePlan(
        topic_hint=topic or None,
        max_tokens=cfg.auto_recall_session_prime_max_tokens,
    )


def plan_mcp_recall_middleware(
    content: str,
    *,
    tool_name: str,
    cfg: ActivationConfig | None,
    auto_observe: bool,
) -> RecallMiddlewarePlan:
    """Plan MCP recall middleware side effects without touching MCP state."""
    should_recall = should_recall_for_tool(tool_name, cfg)
    if not should_recall:
        return RecallMiddlewarePlan(
            should_recall=False,
            surface_notifications_when_recall_disabled=bool(
                tool_name == "get_context"
                and cfg
                and cfg.notification_surfacing_enabled
            ),
            auto_observe_content=False,
            ingest_live_turn=False,
            cache_only=False,
        )

    return RecallMiddlewarePlan(
        should_recall=True,
        surface_notifications_when_recall_disabled=False,
        auto_observe_content=auto_observe and len(content) >= 50,
        ingest_live_turn=tool_name not in WRITE_TOOLS,
        cache_only=tool_name in WRITE_TOOLS,
    )


def compact_auto_recall_surface(
    results: Sequence[Mapping[str, Any]],
    *,
    query: str,
    packets: Sequence[Mapping[str, Any]] | None = None,
    gate: Mapping[str, Any] | None = None,
    min_score: float,
) -> dict[str, Any] | None:
    """Compact raw recall results into the additive MCP auto-recall surface."""
    entities: list[dict[str, Any]] = []
    cue_episodes: list[dict[str, Any]] = []

    for result in results:
        score = _score(result.get("score"))
        if score < min_score:
            continue

        if result.get("result_type") == "entity" and "entity" in result:
            entity = _mapping(result.get("entity"))
            entry: dict[str, Any] = {
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "summary": str(entity.get("summary") or "")[:100],
            }
            top_facts = [
                str(_mapping(relationship).get("predicate", "?"))
                for relationship in (result.get("relationships") or [])[:3]
            ]
            if top_facts:
                entry["top_facts"] = top_facts
            entities.append(entry)
            continue

        if result.get("result_type") == "cue_episode":
            cue = _mapping(result.get("cue"))
            cue_episodes.append(
                {
                    "episode_id": cue.get("episode_id"),
                    "cue_text": str(cue.get("cue_text") or "")[:140],
                    "supporting_spans": list(cue.get("supporting_spans") or [])[:2],
                    "projection_state": cue.get("projection_state"),
                    "score": round(score, 4),
                }
            )

    packet_list = list(packets or [])
    if not entities and not cue_episodes and not packet_list:
        return None

    response: dict[str, Any] = {
        "source": "auto_recall",
        "query_used": query,
        "packets": packet_list,
        "entities": entities,
    }
    if gate:
        response["gate"] = dict(gate)
    if cue_episodes:
        response["cue_episodes"] = cue_episodes
    return response


def compact_lite_auto_recall_surface(
    results: Sequence[Mapping[str, Any]],
    *,
    level: str,
) -> dict[str, Any] | None:
    """Compact lite/medium entity-probe recall into the MCP piggyback surface."""
    if not results:
        return None
    return {
        "source": f"recall_{level}",
        "entities": list(results),
    }


async def _record_auto_recall_gate(
    manager: Any,
    *,
    group_id: str,
    cfg: ActivationConfig,
    profile: str,
    mode: str,
    status: str,
    skip_reason: str | None = None,
    duration_ms: float = 0.0,
    timeout: bool = False,
    cache_hit: bool | None = None,
) -> None:
    budget = recall_budget_for_profile(
        cfg,
        profile,
        surface="mcp",
        mode=mode,
        max_results=cfg.auto_recall_limit,
        max_packets=cfg.recall_packet_auto_limit,
        max_output_tokens=cfg.auto_recall_token_budget,
    )
    budget_miss = budget.exceeded(duration_ms)
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="auto_recall_gate",
            source="auto_recall",
            mode=mode,
            status=status,
            duration_ms=duration_ms,
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            skip_reason=skip_reason,
            timeout=timeout,
            budget_miss=budget_miss,
            degraded=bool(timeout or (budget.timeout_degrades and budget_miss)),
            cache_hit=cache_hit,
        ),
    )


async def _record_packet_cache_operation(
    manager: Any,
    *,
    group_id: str,
    cache_hit: bool,
    packet_count: int,
    duration_ms: float = 0.0,
) -> None:
    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="packet_cache",
            source="auto_recall",
            mode="auto_recall_packet",
            status="ok",
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            packet_count=packet_count,
        ),
    )


def _annotate_memory_need_gate(
    need: Any,
    *,
    mode_requested: str | None = None,
    mode_executed: str | None = None,
    skip_reason: str | None = None,
    budget_profile: str | None = None,
    cache_hit: bool | None = None,
    cache_satisfied: bool = False,
    budget_skipped: bool = False,
) -> None:
    if need is None:
        return
    if mode_requested is not None:
        need.mode_requested = mode_requested
    if mode_executed is not None:
        need.mode_executed = mode_executed
    if skip_reason is not None:
        need.skip_reason = skip_reason
    if budget_profile is not None:
        need.budget_profile = budget_profile
    if cache_hit is not None:
        need.cache_hit = cache_hit
    if cache_satisfied:
        need.cache_satisfied = True
    if budget_skipped:
        need.budget_skipped = True


async def _finish_memory_need_gate(
    manager: Any,
    *,
    group_id: str,
    cfg: ActivationConfig,
    event_bus: Any,
    need: Any,
    content: str,
    mode_executed: str,
    skip_reason: str | None = None,
    cache_hit: bool | None = None,
    cache_satisfied: bool = False,
    budget_skipped: bool = False,
) -> None:
    if need is None:
        return
    _annotate_memory_need_gate(
        need,
        mode_executed=mode_executed,
        skip_reason=skip_reason,
        cache_hit=cache_hit,
        cache_satisfied=cache_satisfied,
        budget_skipped=budget_skipped,
    )
    await record_manager_memory_need_analysis(manager, group_id, need)
    if cfg.recall_telemetry_enabled:
        publish_memory_need_analysis(
            event_bus,
            group_id,
            need,
            source="auto_recall",
            mode="auto_recall",
            turn_text=content,
        )


def _memory_need_gate_payload(need: Any, *, decision: str) -> dict[str, Any]:
    payload = {
        "decision": decision,
        "needType": getattr(need, "need_type", None),
        "modeRequested": getattr(need, "mode_requested", None),
        "modeExecuted": getattr(need, "mode_executed", None),
        "budgetProfile": getattr(need, "budget_profile", None),
    }
    skip_reason = getattr(need, "skip_reason", None)
    if skip_reason:
        payload["skipReason"] = skip_reason
    cache_hit = getattr(need, "cache_hit", None)
    if cache_hit is not None:
        payload["cacheHit"] = cache_hit
    if getattr(need, "cache_satisfied", False):
        payload["cacheSatisfied"] = True
    if getattr(need, "budget_skipped", False):
        payload["budgetSkipped"] = True
    return payload


def _get_cached_packets(
    manager: Any,
    *,
    group_id: str,
    scope: str,
    topic_hint: str,
) -> list[dict[str, Any]] | None:
    get_cached = getattr(manager, "get_cached_memory_packets", None)
    if not callable(get_cached):
        return None
    hit = get_cached(
        group_id,
        scope=scope,
        topic_hint=topic_hint,
        sync_persistent=False,
    )
    if inspect.isawaitable(hit):
        close = getattr(hit, "close", None)
        if callable(close):
            close()
        return None
    return hit.packets if hit is not None else None


def _get_recent_cached_packets_for_query(
    manager: Any,
    *,
    group_id: str,
    query: str,
    max_packets: int,
) -> list[dict[str, Any]]:
    if max_packets <= 0:
        return []
    get_recent = getattr(manager, "get_recent_cached_memory_packets", None)
    if not callable(get_recent):
        return []
    try:
        packets = get_recent(
            group_id,
            scopes=("identity_core", "project_home", EXPLICIT_RECALL_PACKET_CACHE_SCOPE),
            limit_packets=max_packets * 2,
            sync_persistent=False,
        )
    except Exception:
        logger.debug("auto-recall recent packet lookup failed", exc_info=True)
        return []
    if inspect.isawaitable(packets):
        close = getattr(packets, "close", None)
        if callable(close):
            close()
        return []
    if not isinstance(packets, list):
        return []
    return _filter_packets_for_query(
        packets,
        query=query,
        limit=max_packets,
    )


def _cache_packets(
    manager: Any,
    *,
    group_id: str,
    scope: str,
    topic_hint: str,
    packets: Sequence[Mapping[str, Any]],
    build_duration_ms: float,
) -> None:
    cache = getattr(manager, "cache_memory_packets", None)
    if not callable(cache) or inspect.iscoroutinefunction(cache):
        return
    result = cache(
        group_id,
        scope=scope,
        topic_hint=topic_hint,
        packets=packets,
        build_duration_ms=build_duration_ms,
    )
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        return


def _get_packet_feedback_lookup(
    manager: Any,
    group_id: str,
    results: Sequence[dict],
) -> dict[str, dict[str, Any]]:
    memory_ids = _packet_feedback_ids(results)
    if not memory_ids:
        return {}
    getter = getattr(manager, "get_recall_feedback_summary", None)
    if not callable(getter):
        return {}
    lookup = getter(group_id=group_id, memory_ids=memory_ids)
    if inspect.isawaitable(lookup):
        close = getattr(lookup, "close", None)
        if callable(close):
            close()
        return {}
    return dict(lookup or {})


def _packet_feedback_ids(results: Sequence[dict]) -> list[str]:
    memory_ids: list[str] = []
    for result in results:
        result_type = result.get("result_type")
        if result_type == "cue_episode":
            cue = result.get("cue") if isinstance(result.get("cue"), dict) else {}
            episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
            episode_id = cue.get("episode_id") or episode.get("id")
            if episode_id:
                memory_ids.extend([f"cue:{episode_id}", episode_id, f"episode:{episode_id}"])
            continue
        if result_type == "episode":
            episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
            episode_id = episode.get("id")
            if episode_id:
                memory_ids.extend([episode_id, f"episode:{episode_id}"])
            continue
        entity = result.get("entity") if isinstance(result.get("entity"), dict) else {}
        entity_id = entity.get("id")
        if entity_id:
            memory_ids.append(entity_id)
    return list(dict.fromkeys(memory_ids))


def _memory_need_skip_reason(need: Any) -> str:
    reasons = [str(reason) for reason in getattr(need, "reasons", []) or []]
    if "acknowledgement" in reasons:
        return "skipped_ack"
    if "empty_turn" in reasons:
        return "skipped_low_signal"
    if "recent_duplicate" in reasons:
        return "skipped_recent_duplicate"
    if reasons:
        return "skipped_low_signal"
    return "skipped_low_signal"


async def build_lite_auto_recall_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session_cache: MutableMapping[str, Any],
    cfg: ActivationConfig,
    cache_only: bool = False,
) -> dict[str, Any] | None:
    """Dispatch lite/medium MCP auto-recall and compact the additive surface."""
    if len(content) < 20:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(getattr(cfg, "auto_recall_level", "lite")),
            status="skipped",
            skip_reason="skipped_low_signal",
        )
        return None

    level = getattr(cfg, "auto_recall_level", "lite")
    budget = recall_budget_for_profile(
        cfg,
        "auto_lite",
        surface="mcp",
        mode=str(level),
        max_results=5,
        max_packets=cfg.recall_packet_auto_limit,
        max_output_tokens=cfg.auto_recall_token_budget,
    )
    timeout_seconds = budget.stage_timeout_seconds(budget.max_search_ms)
    started = time.perf_counter()
    if timeout_seconds <= 0:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(level),
            status="skipped",
            skip_reason="skipped_budget",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    if cfg.recall_packets_enabled:
        packet_started = time.perf_counter()
        cached_packets = _get_recent_cached_packets_for_query(
            manager,
            group_id=group_id,
            query=content,
            max_packets=cfg.recall_packet_auto_limit,
        )
        packet_duration_ms = round((time.perf_counter() - packet_started) * 1000, 4)
        if cached_packets:
            await _record_packet_cache_operation(
                manager,
                group_id=group_id,
                cache_hit=True,
                packet_count=len(cached_packets),
                duration_ms=packet_duration_ms,
            )
            await _record_auto_recall_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                profile="auto_lite",
                mode=str(level),
                status="ok",
                skip_reason="cache_satisfied",
                duration_ms=round((time.perf_counter() - started) * 1000, 4),
            )
            return compact_auto_recall_surface(
                [],
                query=content,
                packets=cached_packets,
                gate={
                    "decision": "skipped_cache_satisfied",
                    "modeRequested": str(level),
                    "modeExecuted": "cached",
                    "budgetProfile": "auto_lite",
                    "skipReason": "cache_satisfied",
                    "cacheHit": True,
                    "cacheSatisfied": True,
                },
                min_score=cfg.auto_recall_min_score,
            )

    if cache_only:
        await _record_packet_cache_operation(
            manager,
            group_id=group_id,
            cache_hit=False,
            packet_count=0,
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(level),
            status="skipped",
            skip_reason="cache_miss",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
            cache_hit=False,
        )
        return None

    try:
        if level == "medium" and hasattr(manager, "recall_medium"):
            results = await asyncio.wait_for(
                manager.recall_medium(
                    text=content,
                    group_id=group_id,
                    session_cache=session_cache,
                    token_budget=cfg.auto_recall_token_budget,
                    cache_ttl=cfg.auto_recall_cache_ttl_seconds,
                ),
                timeout=timeout_seconds,
            )
        else:
            results = await asyncio.wait_for(
                manager.recall_lite(
                    text=content,
                    group_id=group_id,
                    session_cache=session_cache,
                    token_budget=cfg.auto_recall_token_budget,
                    cache_ttl=cfg.auto_recall_cache_ttl_seconds,
                ),
                timeout=timeout_seconds,
            )
    except TimeoutError:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(level),
            status="degraded",
            skip_reason="recall_timeout",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
            timeout=True,
        )
        return None
    except Exception:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(level),
            status="error",
            skip_reason="error",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    response = compact_lite_auto_recall_surface(results, level=level)
    if response is None:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_lite",
            mode=str(level),
            status="skipped",
            skip_reason="skipped_no_results",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
    return response


async def build_full_auto_recall_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    cfg: ActivationConfig,
    session_last_recall_time: float | None,
    cooldown: RecallCooldown | None,
    event_bus: Any = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Run full MCP auto-recall and compact the additive recall surface."""
    started = time.perf_counter()
    budget = recall_budget_for_profile(
        cfg,
        "auto_deep",
        surface="mcp",
        mode="auto_recall",
        max_results=cfg.auto_recall_limit,
        max_packets=cfg.recall_packet_auto_limit,
        max_output_tokens=cfg.auto_recall_token_budget,
    )
    if not cfg.auto_recall_enabled:
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_disabled",
        )
        return None

    need = None
    query = ""
    conv_context = manager_conversation_context(manager)
    if cfg.recall_need_analyzer_enabled:
        recent_turns = (
            manager_conversation_recent_turns(manager, cfg.conv_multi_query_turns)
            if conv_context is not None
            else []
        )
        session_entity_names = manager_conversation_top_entity_names(manager)
        graph_probe = (
            manager.get_recall_need_graph_probe()
            if cfg.recall_need_graph_probe_enabled
            else None
        )
        need = await analyze_memory_need(
            content,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            mode="auto_recall",
            graph_probe=graph_probe,
            group_id=group_id,
            conv_context=conv_context,
            cfg=cfg,
            thresholds=await resolve_manager_recall_need_thresholds(manager, group_id),
        )
        _annotate_memory_need_gate(
            need,
            mode_requested="deep" if need.should_recall else "none",
            budget_profile=budget.profile,
        )
        if not need.should_recall:
            skip_reason = _memory_need_skip_reason(need)
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason=skip_reason,
            )
            await _record_auto_recall_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                profile="auto_deep",
                mode="auto_recall",
                status="skipped",
                skip_reason=skip_reason,
                duration_ms=round((time.perf_counter() - started) * 1000, 4),
            )
            return None
        query = need.query_hint or extract_recall_query(content)
    else:
        query = extract_recall_query(content)

    if not query:
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason="skipped_low_signal",
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_low_signal",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    current_time = time.time() if now is None else now
    if cooldown and cooldown.is_throttled(query, current_time):
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason="skipped_recent_duplicate",
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_recent_duplicate",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    if session_last_recall_time and (current_time - session_last_recall_time) < 30.0:
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason="skipped_recent_explicit",
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_recent_explicit",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    recall_limit = cfg.auto_recall_limit
    if budget.max_results <= 0 or budget.max_output_tokens <= 0:
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason="skipped_budget",
                budget_skipped=True,
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_budget",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None
    if (
        cfg.conv_topic_shift_enabled
        and conv_context is not None
        and conv_context.detect_topic_shift()
    ):
        recall_limit = cfg.conv_topic_shift_recall_boost
        conv_context.acknowledge_shift()

    try:
        interaction_type = None
        record_access = True
        if cfg.recall_telemetry_enabled or cfg.recall_usage_feedback_enabled:
            interaction_type = "surfaced"
        if cfg.recall_usage_feedback_enabled:
            record_access = False
        cached_packets = None
        if cfg.recall_packets_enabled:
            cached_packets = _get_cached_packets(
                manager,
                group_id=group_id,
                scope="auto_recall_packet",
                topic_hint=query,
            )
            if cached_packets is not None:
                await _record_packet_cache_operation(
                    manager,
                    group_id=group_id,
                    cache_hit=True,
                    packet_count=len(cached_packets),
                )
                if need is not None:
                    await _finish_memory_need_gate(
                        manager,
                        group_id=group_id,
                        cfg=cfg,
                        event_bus=event_bus,
                        need=need,
                        content=content,
                        mode_executed="cached",
                        skip_reason="skipped_cache_satisfied",
                        cache_hit=True,
                        cache_satisfied=True,
                    )
                if cooldown:
                    cooldown.record(query, current_time)
                return compact_auto_recall_surface(
                    [],
                    query=query,
                    packets=cached_packets,
                    gate=(
                        _memory_need_gate_payload(
                            need,
                            decision="skipped_cache_satisfied",
                        )
                        if need is not None
                        else None
                    ),
                    min_score=cfg.auto_recall_min_score,
                )
        results = await manager.recall(
            query=query,
            group_id=group_id,
            limit=recall_limit,
            record_access=record_access,
            interaction_type=interaction_type,
            interaction_source="auto_recall",
            memory_need=need,
        )
    except Exception:
        logger.debug("auto_recall failed", exc_info=True)
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="none",
                skip_reason="error",
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="error",
            skip_reason="error",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    packets = []
    packet_cache_hit: bool | None = None
    if cfg.recall_packets_enabled:
        packet_scope = "auto_recall_packet"
        cached_packets = _get_cached_packets(
            manager,
            group_id=group_id,
            scope=packet_scope,
            topic_hint=query,
        )
        if cached_packets is not None:
            packets = cached_packets
            packet_cache_hit = True
            await _record_packet_cache_operation(
                manager,
                group_id=group_id,
                cache_hit=True,
                packet_count=len(packets),
            )
        else:
            packet_started = time.perf_counter()
            packets = [
                packet.to_dict()
                for packet in await assemble_memory_packets(
                    results,
                    query,
                    mode="auto_surface",
                    memory_need=need,
                    max_packets=cfg.recall_packet_auto_limit,
                    resolve_entity_name=lambda entity_id: manager.resolve_entity_name(
                        entity_id,
                        group_id,
                    ),
                    feedback_lookup=_get_packet_feedback_lookup(manager, group_id, results),
                )
            ]
            packet_duration_ms = round((time.perf_counter() - packet_started) * 1000, 4)
            _cache_packets(
                manager,
                group_id=group_id,
                scope=packet_scope,
                topic_hint=query,
                packets=packets,
                build_duration_ms=packet_duration_ms,
            )
            packet_cache_hit = False
            await _record_packet_cache_operation(
                manager,
                group_id=group_id,
                cache_hit=False,
                packet_count=len(packets),
                duration_ms=packet_duration_ms,
            )

    response = compact_auto_recall_surface(
        results,
        query=query,
        packets=packets,
        gate=(
            _memory_need_gate_payload(need, decision="triggered")
            if need is not None
            else None
        ),
        min_score=cfg.auto_recall_min_score,
    )
    if response is None:
        if need is not None:
            await _finish_memory_need_gate(
                manager,
                group_id=group_id,
                cfg=cfg,
                event_bus=event_bus,
                need=need,
                content=content,
                mode_executed="deep",
                skip_reason="skipped_no_results",
                cache_hit=packet_cache_hit,
            )
        await _record_auto_recall_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            profile="auto_deep",
            mode="auto_recall",
            status="skipped",
            skip_reason="skipped_no_results",
            duration_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        return None

    if need is not None:
        await _finish_memory_need_gate(
            manager,
            group_id=group_id,
            cfg=cfg,
            event_bus=event_bus,
            need=need,
            content=content,
            mode_executed="deep",
            cache_hit=packet_cache_hit,
        )
        response["gate"] = _memory_need_gate_payload(need, decision="triggered")

    if cooldown:
        cooldown.record(query, current_time)

    return response


async def build_session_prime_surface(
    manager: Any,
    *,
    content: str | None,
    group_id: str,
    cfg: ActivationConfig,
    already_primed: bool,
) -> SessionPrimeSurface:
    """Fetch first-call MCP session context through a route-neutral helper."""
    plan = plan_session_prime(content, cfg, already_primed=already_primed)
    if plan is None:
        return SessionPrimeSurface(context=None, should_mark_primed=False)

    budget = recall_budget_for_profile(
        cfg,
        "startup",
        surface="mcp",
        mode="mcp_session_prime",
        max_packets=cfg.recall_packet_auto_limit,
        max_output_tokens=plan.max_tokens,
    )
    started = time.perf_counter()
    packets = _session_prime_cached_packets(
        manager,
        group_id=group_id,
        topic_hint=plan.topic_hint,
        limit_packets=cfg.recall_packet_auto_limit,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 4)
    budget_miss = budget.exceeded(duration_ms)
    degraded = bool(budget.timeout_degrades and budget_miss)

    if packets:
        context = MemoryContextBuilder.render_cached_packets(packets)
        await record_manager_memory_operation(
            manager,
            group_id,
            MemoryOperationSample(
                operation="context",
                source="mcp_session_prime",
                mode="mcp_session_prime",
                status="ok",
                duration_ms=duration_ms,
                budget_ms=budget.budget_ms,
                budget_tokens=budget.budget_tokens,
                budget_miss=budget_miss,
                degraded=degraded,
                cache_hit=True,
                packet_count=len(packets),
            ),
        )
        return SessionPrimeSurface(
            context={
                "context": context,
                "entity_count": 0,
                "fact_count": 0,
                "token_estimate": MemoryContextBuilder.estimate_tokens(context),
                "format": "structured",
                "cached_packets": packets,
                "packet_cache": {
                    "hit": True,
                    "packet_count": len(packets),
                    "scopes": _packet_scope_counts(packets),
                },
                "status": "ok",
                "budget": {
                    **budget.to_dict(),
                    "duration_ms": duration_ms,
                    "budget_miss": budget_miss,
                    "timeout": False,
                    "degraded": degraded,
                    "skip_reason": None,
                },
                "lifecycle": {
                    "stage": "recall",
                    "degraded": degraded,
                    "timeout": False,
                    "skip_reason": None,
                },
                "diagnostics": {"stage_timings_ms": {"packet_cache": duration_ms}},
            },
            should_mark_primed=True,
        )

    await record_manager_memory_operation(
        manager,
        group_id,
        MemoryOperationSample(
            operation="context",
            source="mcp_session_prime",
            mode="mcp_session_prime",
            status="skipped",
            duration_ms=duration_ms,
            skip_reason="cache_miss",
            budget_ms=budget.budget_ms,
            budget_tokens=budget.budget_tokens,
            budget_miss=budget_miss,
            degraded=degraded,
            cache_hit=False,
            packet_count=0,
        ),
    )
    return SessionPrimeSurface(context=None, should_mark_primed=True)


def _session_prime_cached_packets(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    limit_packets: int,
) -> list[dict[str, Any]]:
    """Load startup-safe cached packets without invoking graph or recall paths."""
    limit = max(1, int(limit_packets or 1))
    packets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    get_cached_packets = getattr(manager, "get_cached_memory_packets", None)
    if callable(get_cached_packets):
        for scope, scope_topic in (
            ("identity_core", None),
            ("project_home", topic_hint),
            ("project_home", None),
        ):
            if len(packets) >= limit:
                break
            try:
                hit = get_cached_packets(
                    group_id,
                    scope=scope,
                    topic_hint=scope_topic,
                    project_path=None,
                    sync_persistent=False,
                )
            except Exception:
                logger.debug("session_prime packet-cache lookup failed", exc_info=True)
                continue
            if inspect.isawaitable(hit):
                _close_awaitable(hit)
                continue
            hit_packets = getattr(hit, "packets", None)
            if not isinstance(hit_packets, list):
                continue
            for packet in hit_packets:
                if len(packets) >= limit:
                    break
                if isinstance(packet, Mapping):
                    _append_unique_session_packet(packets, seen, packet, scope=scope)

    if len(packets) >= limit:
        return packets

    get_recent = getattr(manager, "get_recent_cached_memory_packets", None)
    if not callable(get_recent):
        return packets
    try:
        recent = get_recent(
            group_id,
            scopes=("identity_core", "project_home"),
            limit_packets=limit,
            sync_persistent=False,
        )
    except Exception:
        logger.debug("session_prime recent packet lookup failed", exc_info=True)
        return packets
    if inspect.isawaitable(recent):
        _close_awaitable(recent)
        return packets
    if not isinstance(recent, list):
        return packets
    for packet in recent:
        if len(packets) >= limit:
            break
        if isinstance(packet, Mapping):
            scope = str(packet.get("_cache_scope") or packet.get("packet_type") or "recent")
            _append_unique_session_packet(packets, seen, packet, scope=scope)
    return packets


def _append_unique_session_packet(
    packets: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    packet: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    payload = {**dict(packet), "_cache_scope": scope}
    key = (
        str(payload.get("packet_type") or ""),
        str(payload.get("title") or ""),
        str(payload.get("summary") or ""),
    )
    if key in seen:
        return
    seen.add(key)
    packets.append(payload)


def _packet_scope_counts(packets: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for packet in packets:
        scope = str(packet.get("_cache_scope") or packet.get("packet_type") or "unknown")
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _close_awaitable(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


async def store_mcp_auto_observe_turn(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str = "tool_piggyback",
) -> None:
    """Store MCP auto-observed content through the Capture-stage helper boundary."""
    try:
        await store_observation(
            manager,
            content=content,
            group_id=group_id,
            source=source,
        )
    except Exception:
        logger.debug("middleware auto-observe failed", exc_info=True)


async def drain_mcp_triggered_intentions(manager: Any) -> list[dict] | None:
    """Drain triggered prospective-memory intentions for MCP recall enrichment."""
    drain = getattr(manager, "drain_triggered_intention_views", None)
    if not callable(drain):
        return None
    result = drain()
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, list) and result else None


async def run_mcp_recall_middleware(
    response: MutableMapping[str, Any],
    *,
    content: str,
    tool_name: str,
    cfg: ActivationConfig | None,
    group_id: str,
    get_manager: Callable[[], Any],
    load_notifications: Callable[[ActivationConfig, str], Sequence[Mapping[str, Any]] | None],
    auto_recall_lite: Callable[..., Awaitable[Mapping[str, Any] | None]],
    session_prime: Callable[
        [str | None, Any, ActivationConfig],
        Awaitable[Mapping[str, Any] | None],
    ],
    ingest_live_turn: Callable[[Any, str], Awaitable[None]],
    auto_observe: bool = False,
) -> None:
    """Execute MCP recall middleware side effects behind a retrieval boundary."""
    plan = plan_mcp_recall_middleware(
        content,
        tool_name=tool_name,
        cfg=cfg,
        auto_observe=auto_observe,
    )
    if not plan.should_recall:
        if plan.surface_notifications_when_recall_disabled and cfg:
            apply_mcp_recall_enrichment(
                response,
                memory_notifications=load_notifications(cfg, group_id),
            )
        return

    assert cfg is not None
    manager = get_manager()

    if plan.auto_observe_content:
        await store_mcp_auto_observe_turn(
            manager,
            content=content,
            group_id=group_id,
        )

    if plan.ingest_live_turn:
        await ingest_live_turn(manager, content)

    prime = await session_prime(content, manager, cfg)
    recalled = await auto_recall_lite(content, manager, cfg, cache_only=plan.cache_only)
    intentions = await drain_mcp_triggered_intentions(manager)
    notifications = load_notifications(cfg, group_id)

    apply_mcp_recall_enrichment(
        response,
        session_context=prime,
        recalled_context=recalled,
        triggered_intentions=intentions,
        memory_notifications=notifications,
    )


def apply_mcp_recall_enrichment(
    response: MutableMapping[str, Any],
    *,
    session_context: Mapping[str, Any] | None = None,
    recalled_context: Mapping[str, Any] | None = None,
    triggered_intentions: Sequence[Mapping[str, Any]] | None = None,
    memory_notifications: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Attach non-empty recall enrichment fields to an MCP response."""
    if session_context:
        response["session_context"] = dict(session_context)
    if recalled_context:
        response["recalled_context"] = dict(recalled_context)
    if triggered_intentions:
        response["triggered_intentions"] = [dict(item) for item in triggered_intentions]
    if memory_notifications:
        response["memory_notifications"] = [dict(item) for item in memory_notifications]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _score(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
