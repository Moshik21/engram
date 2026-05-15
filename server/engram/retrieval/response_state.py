"""Recall response-state helpers for public MCP/REST enrichment."""

from __future__ import annotations

from typing import Any

from engram.storage.protocols import ActivationStore


class RecallResponseStateService:
    """Format transient recall-side state without exposing manager internals."""

    def triggered_intention_views(self, matches: list[Any]) -> list[dict[str, Any]]:
        """Serialize triggered prospective-memory matches for MCP piggybacking."""
        return [
            {
                "trigger": match.trigger_text,
                "action": match.action_text,
                "similarity": round(match.similarity, 4),
                "matched_via": match.matched_via,
                **({"context": match.context} if match.context else {}),
                **({"see_also": match.see_also} if match.see_also else {}),
            }
            for match in matches
        ]

    def near_miss_views(self, near_misses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a defensive copy of the latest near-miss payloads."""
        return list(near_misses)

    async def get_access_count(
        self,
        activation_store: ActivationStore,
        entity_id: str,
    ) -> int:
        """Read an entity access count for recall presentation."""
        if not entity_id:
            return 0
        state = await activation_store.get_activation(entity_id)
        return state.access_count if state else 0

    def surprise_connection_views(
        self,
        surprise_cache: Any,
        *,
        group_id: str,
        now: float,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Serialize cached surprise connections for public response payloads."""
        if surprise_cache is None or not hasattr(surprise_cache, "get"):
            return []
        surprises = surprise_cache.get(group_id, now)
        if not surprises:
            return []
        return [
            {
                "entity": surprise.entity_name,
                "connected_to": surprise.connected_to_name,
                "relationship": surprise.predicate,
                "surprise_score": round(surprise.surprise_score, 4),
            }
            for surprise in surprises[:limit]
        ]
