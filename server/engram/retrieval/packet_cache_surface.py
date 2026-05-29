"""Operator surfaces for packet-cache diagnostics and maintenance."""

from __future__ import annotations

from typing import Any


def build_api_packet_cache_summary_surface(
    manager: Any,
    *,
    group_id: str,
) -> dict[str, Any]:
    """Return packet-cache diagnostics without mutating cache state."""
    summary = {}
    get_summary = getattr(manager, "get_memory_packet_cache_summary", None)
    if callable(get_summary):
        summary = get_summary(group_id) or {}
    return {
        "operation": "packet_cache.summary",
        "status": "ok",
        "packetCache": _camel_packet_cache_summary(summary),
    }


def load_packet_cache_summary(
    manager: Any,
    *,
    group_id: str,
) -> dict[str, Any]:
    """Return the raw packet-cache summary mapping without mutating cache state."""
    get_summary = getattr(manager, "get_memory_packet_cache_summary", None)
    if not callable(get_summary):
        return {}
    try:
        summary = get_summary(group_id)
    except Exception:
        return {}
    return summary if isinstance(summary, dict) else {}


def build_api_packet_cache_clear_surface(
    manager: Any,
    *,
    group_id: str,
) -> dict[str, Any]:
    """Clear packet-cache entries for the current tenant group."""
    cleared_count = int(manager.clear_memory_packet_cache(group_id) or 0)
    summary = {}
    get_summary = getattr(manager, "get_memory_packet_cache_summary", None)
    if callable(get_summary):
        summary = get_summary(group_id) or {}
    return {
        "operation": "packet_cache.clear",
        "status": "cleared",
        "clearedCount": cleared_count,
        "packetCache": _camel_packet_cache_summary(summary),
    }


def _camel_packet_cache_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the packet-cache summary in the REST/AXI camelCase shape."""
    return {
        "entryCount": int(summary.get("entry_count") or summary.get("entryCount") or 0),
        "freshCount": int(summary.get("fresh_count") or summary.get("freshCount") or 0),
        "invalidatedCount": int(
            summary.get("invalidated_count") or summary.get("invalidatedCount") or 0
        ),
        "expiredCount": int(summary.get("expired_count") or summary.get("expiredCount") or 0),
        "hitCount": int(summary.get("hit_count") or summary.get("hitCount") or 0),
        "scopes": summary.get("scopes") or {},
        "persistent": bool(summary.get("persistent")),
        "path": summary.get("path"),
    }
