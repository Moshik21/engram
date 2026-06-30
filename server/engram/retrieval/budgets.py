"""Shared recall budget profiles for memory value instrumentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecallBudget:
    """Budget applied to a recall/context operation."""

    profile: str
    surface: str
    mode: str
    max_wall_ms: int
    max_search_ms: int
    max_graph_ms: int
    max_packet_ms: int
    max_results: int
    max_packets: int
    max_output_tokens: int
    allow_deep_recall: bool
    allow_embeddings: bool
    allow_graph_probe: bool
    allow_cache_only: bool = False
    timeout_degrades: bool = True
    started_at: float = 0.0

    @classmethod
    def start(cls, **kwargs: Any) -> RecallBudget:
        """Create a budget with an active monotonic start timestamp."""
        return cls(started_at=time.perf_counter(), **kwargs)

    @property
    def budget_ms(self) -> int:
        """Compatibility alias for memory operation samples."""
        return self.max_wall_ms

    @property
    def budget_tokens(self) -> int:
        """Compatibility alias for memory operation samples."""
        return self.max_output_tokens

    def elapsed_ms(self, now: float | None = None) -> float:
        if self.started_at <= 0:
            return 0.0
        return ((time.perf_counter() if now is None else now) - self.started_at) * 1000.0

    def exceeded(self, duration_ms: float | None = None) -> bool:
        value = self.elapsed_ms() if duration_ms is None else duration_ms
        return self.max_wall_ms > 0 and value > self.max_wall_ms

    def remaining_ms(self, stage_ms: int | float | None = None) -> float:
        """Return the remaining wall/stage budget in milliseconds."""
        stage_budget = float(stage_ms or 0)
        if self.max_wall_ms <= 0:
            return max(0.0, stage_budget)
        wall_remaining = max(0.0, float(self.max_wall_ms) - self.elapsed_ms())
        if stage_budget <= 0:
            return wall_remaining
        return min(wall_remaining, max(0.0, stage_budget))

    def stage_timeout_seconds(self, stage_ms: int | float | None) -> float:
        """Return a bounded timeout for an optional stage."""
        return self.remaining_ms(stage_ms) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "surface": self.surface,
            "mode": self.mode,
            "max_wall_ms": self.max_wall_ms,
            "max_search_ms": self.max_search_ms,
            "max_graph_ms": self.max_graph_ms,
            "max_packet_ms": self.max_packet_ms,
            "max_results": self.max_results,
            "max_packets": self.max_packets,
            "max_output_tokens": self.max_output_tokens,
            "allow_deep_recall": self.allow_deep_recall,
            "allow_embeddings": self.allow_embeddings,
            "allow_graph_probe": self.allow_graph_probe,
            "allow_cache_only": self.allow_cache_only,
            "timeout_degrades": self.timeout_degrades,
        }


def recall_budget_for_profile(
    cfg: Any,
    profile: str,
    *,
    surface: str = "runtime",
    mode: str | None = None,
    max_results: int | None = None,
    max_packets: int | None = None,
    max_output_tokens: int | None = None,
) -> RecallBudget:
    """Return a conservative recall budget for a named profile."""
    timeout_degrades = bool(getattr(cfg, "recall_budget_timeout_degrades", True))
    token_budget = _int(max_output_tokens, getattr(cfg, "auto_recall_token_budget", 300))
    if profile == "startup":
        return RecallBudget.start(
            profile=profile,
            surface=surface,
            mode=mode or "cached",
            max_wall_ms=_int(None, getattr(cfg, "recall_budget_startup_ms", 250)),
            max_search_ms=0,
            max_graph_ms=0,
            max_packet_ms=50,
            max_results=_int(max_results, 0),
            max_packets=_int(max_packets, 0),
            max_output_tokens=token_budget,
            allow_deep_recall=False,
            allow_embeddings=False,
            allow_graph_probe=False,
            allow_cache_only=True,
            timeout_degrades=timeout_degrades,
        )
    if profile == "auto_lite":
        return RecallBudget.start(
            profile=profile,
            surface=surface,
            mode=mode or str(getattr(cfg, "auto_recall_level", "lite")),
            max_wall_ms=_int(None, getattr(cfg, "recall_budget_auto_lite_ms", 350)),
            max_search_ms=200,
            max_graph_ms=150,
            max_packet_ms=75,
            max_results=_int(max_results, 5),
            max_packets=_int(max_packets, 0),
            max_output_tokens=token_budget,
            allow_deep_recall=False,
            allow_embeddings=mode == "medium",
            allow_graph_probe=False,
            timeout_degrades=timeout_degrades,
        )
    if profile == "auto_deep":
        return RecallBudget.start(
            profile=profile,
            surface=surface,
            mode=mode or "deep",
            max_wall_ms=_int(None, getattr(cfg, "recall_budget_auto_deep_ms", 750)),
            max_search_ms=400,
            max_graph_ms=250,
            max_packet_ms=100,
            max_results=_int(max_results, getattr(cfg, "auto_recall_limit", 3)),
            max_packets=_int(max_packets, getattr(cfg, "recall_packet_auto_limit", 0)),
            max_output_tokens=token_budget,
            allow_deep_recall=True,
            allow_embeddings=True,
            allow_graph_probe=bool(getattr(cfg, "recall_need_graph_probe_enabled", False)),
            timeout_degrades=timeout_degrades,
        )
    if profile == "chat":
        return RecallBudget.start(
            profile=profile,
            surface=surface,
            mode=mode or "deep",
            max_wall_ms=_int(None, getattr(cfg, "recall_budget_chat_ms", 1200)),
            max_search_ms=700,
            max_graph_ms=350,
            max_packet_ms=150,
            max_results=_int(max_results, 5),
            max_packets=_int(max_packets, 2),
            max_output_tokens=token_budget,
            allow_deep_recall=True,
            allow_embeddings=True,
            allow_graph_probe=bool(getattr(cfg, "recall_need_graph_probe_enabled", False)),
            timeout_degrades=timeout_degrades,
        )
    return RecallBudget.start(
        profile="explicit",
        surface=surface,
        mode=mode or "deep",
        max_wall_ms=_int(None, getattr(cfg, "recall_budget_explicit_ms", 2000)),
        max_search_ms=_int(None, getattr(cfg, "recall_budget_explicit_search_ms", 900)),
        max_graph_ms=600,
        max_packet_ms=200,
        max_results=_int(max_results, 10),
        max_packets=_int(max_packets, 3),
        max_output_tokens=token_budget,
        allow_deep_recall=True,
        allow_embeddings=True,
        allow_graph_probe=bool(getattr(cfg, "recall_need_graph_probe_enabled", False)),
        timeout_degrades=timeout_degrades,
    )


def budget_profile_for_source(source: str) -> str:
    """Infer a budget profile from a memory operation source label."""
    if source in {"mcp_session_prime", "axi_home"}:
        return "startup"
    if source == "auto_recall":
        return "auto_deep"
    if source in {"mcp_auto_recall", "recall_lite", "recall_medium"}:
        return "auto_lite"
    if source in {"chat", "chat_recall"}:
        return "chat"
    return "explicit"


def surface_for_source(source: str) -> str:
    """Infer a public surface from a source label."""
    if source.startswith("axi"):
        return "axi"
    if source.startswith("mcp") or source == "auto_recall":
        return "mcp"
    if source.startswith("api"):
        return "rest"
    if source.startswith("chat"):
        return "chat"
    return "runtime"


def _int(value: Any, default: Any) -> int:
    try:
        if value is None:
            value = default
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
