"""Fast entity-probe recall for lightweight MCP/REST context."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from engram.models.entity import Entity
from engram.storage.protocols import GraphStore, SearchIndex
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)

GraphProvider = Callable[[], GraphStore | None]
SearchProvider = Callable[[], SearchIndex | None]
EntityNameResolver = Callable[[str, str], Awaitable[str]]


def freshness_label(updated_at: datetime | None) -> str:
    """Compute freshness label from an entity timestamp."""
    if not updated_at:
        return "unknown"
    age = (utc_now() - updated_at).days
    if age <= 7:
        return "fresh"
    if age <= 30:
        return "recent"
    if age <= 90:
        return "aging"
    return "stale"


class EntityProbeRecallService:
    """Own the lightweight entity-probe recall variants used during conversation."""

    _RE_PROPER_NOUNS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    _RE_QUOTED = re.compile(r'"([^"]{2,})"')
    _RE_AT_MENTION = re.compile(r"@(\w+)")
    _RE_HASHTAG = re.compile(r"#(\w+)")
    _RE_ALL_CAPS = re.compile(r"\b[A-Z]{2,}\b")

    def __init__(
        self,
        *,
        get_graph_store: GraphProvider,
        get_search_index: SearchProvider,
        resolve_entity_name: EntityNameResolver,
    ) -> None:
        self._get_graph_store = get_graph_store
        self._get_search_index = get_search_index
        self._resolve_entity_name = resolve_entity_name

    async def recall_lite(
        self,
        text: str,
        group_id: str,
        session_cache: dict[str, tuple[float, dict]] | None = None,
        token_budget: int = 300,
        cache_ttl: float = 300.0,
    ) -> list[dict]:
        """Fast entity-probe recall that avoids embedding and activation work."""
        graph = self._get_graph_store()
        if not text or not text.strip() or graph is None:
            return []

        unique_mentions = self.extract_mentions(text)
        if not unique_mentions:
            return []

        now = time.time()
        if session_cache is None:
            session_cache = {}

        tokens_per_entity = 40
        identity_core_results: list[dict] = []
        normal_results: list[dict] = []

        for mention in unique_mentions:
            candidates = await graph.find_entity_candidates(mention, group_id, limit=3)
            if not candidates:
                continue

            entity = candidates[0]
            cached_result = self._get_cached_result(entity.id, session_cache, now, cache_ttl)
            if cached_result is not None:
                self._append_by_identity(cached_result, identity_core_results, normal_results)
                continue

            result_dict = await self._build_result_dict(entity, group_id, graph)
            session_cache[entity.id] = (now, result_dict)
            self._append_by_identity(result_dict, identity_core_results, normal_results)

        return self._pack_results(
            identity_core_results,
            normal_results,
            token_budget=token_budget,
            tokens_per_entity=tokens_per_entity,
        )

    async def recall_medium(
        self,
        text: str,
        group_id: str,
        session_cache: dict[str, tuple[float, dict]] | None = None,
        token_budget: int = 300,
        cache_ttl: float = 300.0,
        fts_weight: float = 0.3,
        vec_weight: float = 0.7,
    ) -> list[dict]:
        """FTS candidates with optional embedding rerank for fast disambiguation."""
        graph = self._get_graph_store()
        if not text or not text.strip() or graph is None:
            return []

        unique_mentions = self.extract_mentions(text)
        if not unique_mentions:
            return []

        now = time.time()
        if session_cache is None:
            session_cache = {}

        all_candidates: list[tuple[str, Entity, int]] = []
        seen_entity_ids: set[str] = set()
        for mention in unique_mentions:
            candidates = await graph.find_entity_candidates(mention, group_id, limit=3)
            for rank, entity in enumerate(candidates):
                if entity.id not in seen_entity_ids:
                    seen_entity_ids.add(entity.id)
                    all_candidates.append((mention, entity, rank))

        if not all_candidates:
            return []

        sim_scores: dict[str, float] = {}
        search = self._get_search_index()
        if search is not None:
            try:
                entity_ids = [eid for eid in seen_entity_ids]
                sim_scores = await search.compute_similarity(text, entity_ids, group_id)
            except Exception:
                logger.debug("recall_medium embedding rerank failed", exc_info=True)

        scored: list[tuple[float, str, Entity]] = []
        for _mention, entity, fts_rank in all_candidates:
            fts_score = 1.0 / (1 + fts_rank)
            emb_score = sim_scores.get(entity.id, 0.0)
            final = fts_weight * fts_score + vec_weight * emb_score
            scored.append((final, entity.id, entity))

        scored.sort(key=lambda item: item[0], reverse=True)

        tokens_per_entity = 40
        identity_core_results: list[dict] = []
        normal_results: list[dict] = []

        for _score, entity_id, entity in scored:
            cached_result = self._get_cached_result(entity_id, session_cache, now, cache_ttl)
            if cached_result is not None:
                self._append_by_identity(cached_result, identity_core_results, normal_results)
                continue

            result_dict = await self._build_result_dict(entity, group_id, graph)
            session_cache[entity_id] = (now, result_dict)
            self._append_by_identity(result_dict, identity_core_results, normal_results)

        return self._pack_results(
            identity_core_results,
            normal_results,
            token_budget=token_budget,
            tokens_per_entity=tokens_per_entity,
        )

    @classmethod
    def extract_mentions(cls, text: str) -> list[str]:
        """Extract unique entity-like mentions while preserving first-seen order."""
        mentions: list[str] = []
        mentions.extend(cls._RE_PROPER_NOUNS.findall(text))
        mentions.extend(cls._RE_QUOTED.findall(text))
        mentions.extend(cls._RE_AT_MENTION.findall(text))
        mentions.extend(cls._RE_HASHTAG.findall(text))
        mentions.extend(cls._RE_ALL_CAPS.findall(text))

        seen_lower: set[str] = set()
        unique_mentions: list[str] = []
        for mention in mentions:
            key = mention.strip().lower()
            if key and key not in seen_lower and len(key) >= 2:
                seen_lower.add(key)
                unique_mentions.append(mention.strip())
        return unique_mentions

    async def _build_result_dict(
        self,
        entity: Entity,
        group_id: str,
        graph: GraphStore,
    ) -> dict:
        rels = await graph.get_relationships(entity.id, group_id=group_id)
        top_facts: list[str] = []
        for rel in rels[:3]:
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            other_name = await self._resolve_entity_name(other_id, group_id)
            if rel.source_id == entity.id:
                top_facts.append(f"{rel.predicate} {other_name}")
            else:
                top_facts.append(f"{other_name} {rel.predicate}")

        attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
        mat_tier = attrs.get("mat_tier", "episodic")
        if mat_tier == "semantic":
            confidence = "known"
        elif mat_tier == "transitional":
            confidence = "likely"
        else:
            confidence = "recent"

        return {
            "name": entity.name,
            "type": entity.entity_type,
            "summary": (entity.summary or "")[:120],
            "confidence": confidence,
            "identity_core": bool(getattr(entity, "identity_core", False)),
            "top_facts": top_facts,
            "freshness": freshness_label(getattr(entity, "updated_at", None)),
        }

    @staticmethod
    def _get_cached_result(
        entity_id: str,
        session_cache: dict[str, tuple[float, dict]],
        now: float,
        cache_ttl: float,
    ) -> dict | None:
        if entity_id not in session_cache:
            return None
        ts, cached_result = session_cache[entity_id]
        if now - ts < cache_ttl:
            return cached_result
        return None

    @staticmethod
    def _append_by_identity(
        result_dict: dict,
        identity_core_results: list[dict],
        normal_results: list[dict],
    ) -> None:
        if result_dict.get("identity_core"):
            identity_core_results.append(result_dict)
        else:
            normal_results.append(result_dict)

    @staticmethod
    def _pack_results(
        identity_core_results: list[dict],
        normal_results: list[dict],
        *,
        token_budget: int,
        tokens_per_entity: int,
    ) -> list[dict]:
        results: list[dict] = list(identity_core_results)
        remaining_budget = token_budget

        for entry in normal_results:
            if remaining_budget < tokens_per_entity:
                break
            results.append(entry)
            remaining_budget -= tokens_per_entity

        if remaining_budget < 0:
            for entry in results:
                if not entry.get("identity_core"):
                    entry["summary"] = (entry.get("summary") or "")[:60]

        return results
