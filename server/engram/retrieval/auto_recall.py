"""Shared auto-recall policy helpers."""

from __future__ import annotations

import inspect
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig
from engram.retrieval.context import (
    manager_conversation_context,
    manager_conversation_recent_turns,
    manager_conversation_top_entity_names,
)
from engram.retrieval.control import (
    record_manager_memory_need_analysis,
    resolve_manager_recall_need_thresholds,
)
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets

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
        )

    return RecallMiddlewarePlan(
        should_recall=True,
        surface_notifications_when_recall_disabled=False,
        auto_observe_content=auto_observe and len(content) >= 50,
        ingest_live_turn=tool_name not in WRITE_TOOLS,
    )


def compact_auto_recall_surface(
    results: Sequence[Mapping[str, Any]],
    *,
    query: str,
    packets: Sequence[Mapping[str, Any]] | None = None,
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

    if not entities and not cue_episodes:
        return None

    response: dict[str, Any] = {
        "source": "auto_recall",
        "query_used": query,
        "packets": list(packets or []),
        "entities": entities,
    }
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


async def build_lite_auto_recall_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session_cache: MutableMapping[str, Any],
    cfg: ActivationConfig,
) -> dict[str, Any] | None:
    """Dispatch lite/medium MCP auto-recall and compact the additive surface."""
    if len(content) < 20:
        return None

    level = getattr(cfg, "auto_recall_level", "lite")
    try:
        if level == "medium" and hasattr(manager, "recall_medium"):
            results = await manager.recall_medium(
                text=content,
                group_id=group_id,
                session_cache=session_cache,
                token_budget=cfg.auto_recall_token_budget,
                cache_ttl=cfg.auto_recall_cache_ttl_seconds,
            )
        else:
            results = await manager.recall_lite(
                text=content,
                group_id=group_id,
                session_cache=session_cache,
                token_budget=cfg.auto_recall_token_budget,
                cache_ttl=cfg.auto_recall_cache_ttl_seconds,
            )
    except Exception:
        return None

    return compact_lite_auto_recall_surface(results, level=level)


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
    if not cfg.auto_recall_enabled:
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
        if not need.should_recall:
            return None
        query = need.query_hint or extract_recall_query(content)
    else:
        query = extract_recall_query(content)

    if not query:
        return None

    current_time = time.time() if now is None else now
    if cooldown and cooldown.is_throttled(query, current_time):
        return None

    if session_last_recall_time and (current_time - session_last_recall_time) < 30.0:
        return None

    recall_limit = cfg.auto_recall_limit
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
        return None

    packets = []
    if cfg.recall_packets_enabled:
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
            )
        ]

    response = compact_auto_recall_surface(
        results,
        query=query,
        packets=packets,
        min_score=cfg.auto_recall_min_score,
    )
    if response is None:
        return None

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

    try:
        result = await manager.get_context(
            group_id=group_id,
            max_tokens=plan.max_tokens,
            topic_hint=plan.topic_hint,
            format="structured",
        )
        return SessionPrimeSurface(context=result, should_mark_primed=True)
    except Exception:
        logger.debug("session_prime failed", exc_info=True)
        return SessionPrimeSurface(context=None, should_mark_primed=True)


async def store_mcp_auto_observe_turn(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str = "tool_piggyback",
) -> None:
    """Store MCP auto-observed content through the runtime helper boundary."""
    try:
        await manager.store_episode(content, group_id, source=source)
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
    auto_recall_lite: Callable[[str, Any, ActivationConfig], Awaitable[Mapping[str, Any] | None]],
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
    recalled = await auto_recall_lite(content, manager, cfg)
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
