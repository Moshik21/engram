"""Shared auto-recall policy helpers."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig

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
