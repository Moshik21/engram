"""Direct entity and fact lookup services."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.relationship import Relationship

EPISTEMIC_FACT_PREDICATES = {
    "DECIDED_IN",
    "DOCUMENTED_IN",
    "IMPLEMENTED_BY",
    "ANNOUNCED_AS",
    "SUPERSEDED_BY",
}

MCP_SEARCH_ENTITIES_DEPRECATION = (
    "Deprecated compat alias. Use recall(query=...) for retrieval; "
    "pass lookup_kind='entities' with name or entity_type when needed."
)
MCP_SEARCH_FACTS_DEPRECATION = (
    "Deprecated compat alias. Use recall(query=...) for retrieval; "
    "pass lookup_kind='facts' with subject or predicate when needed."
)


async def _prime_recall_lookup_path(
    manager: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
    cfg: Any,
    project_path: str | None = None,
) -> dict[str, Any] | None:
    """Run explicit recall before lookup compat shims (no transport middleware)."""
    if not query.strip():
        return None
    from engram.retrieval.recall_surface import build_mcp_recall_surface

    return await build_mcp_recall_surface(
        manager,
        group_id=group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        project_path=project_path,
    )


def _attach_lookup_compat_metadata(
    result: dict[str, Any],
    *,
    tool_name: str,
    recall_companion: dict[str, Any] | None,
    recall_query: str,
) -> None:
    result["preferRecall"] = True
    result["deprecationNotice"] = (
        MCP_SEARCH_ENTITIES_DEPRECATION
        if tool_name == "search_entities"
        else MCP_SEARCH_FACTS_DEPRECATION
    )
    if recall_companion is not None:
        result["recallCompanion"] = {
            "query": recall_query,
            "total": recall_companion.get("total", 0),
            "resultCount": len(recall_companion.get("results") or []),
        }


async def build_api_entity_search_surface(
    manager: Any,
    *,
    group_id: str,
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 20,
) -> dict:
    """Build the REST entity-search response from the shared lookup facade."""
    try:
        results = await manager.search_entities(
            group_id=group_id,
            name=name,
            entity_type=entity_type,
            limit=limit,
        )
    except Exception as exc:
        # Silent-inert hardening: type-only listings full-scan the native
        # store and can time out on large brains. That used to surface as an
        # EMPTY 200 ("no Decisions exist") — now the degradation is explicit
        # so callers (e.g. the organic continuity gate) know to use their
        # indexed-probe fallback instead of trusting an empty list.
        if type(exc).__name__ != "NativeQueryError":
            raise
        return {
            "items": [],
            "total": 0,
            "status": "timeout" if getattr(exc, "timeout", False) else "error",
            "detail": str(exc)[:200],
        }
    items = [_api_entity_item(result) for result in results]
    return {"items": items, "total": len(items)}


async def build_mcp_entity_search_surface(
    manager: Any,
    *,
    group_id: str,
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 10,
) -> dict:
    """Build the MCP entity-search response while preserving MCP validation."""
    if not name and not entity_type:
        return {
            "status": "error",
            "message": "At least one of 'name' or 'entity_type' is required.",
        }
    entities = await manager.search_entities(
        group_id=group_id,
        name=name,
        entity_type=entity_type,
        limit=limit,
    )
    return {"entities": entities, "total": len(entities)}


async def build_mcp_entity_search_tool_surface(
    manager: Any,
    *,
    group_id: str,
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 10,
    recall_middleware: Callable[..., Awaitable[None]],
    cfg: Any | None = None,
    project_path: str | None = None,
) -> dict:
    """Build the MCP entity-search compat shim and prime the unified recall path."""
    recall_query = name or entity_type or ""
    recall_companion = None
    if cfg is not None:
        recall_companion = await _prime_recall_lookup_path(
            manager,
            group_id=group_id,
            query=recall_query,
            limit=limit,
            cfg=cfg,
            project_path=project_path,
        )
    result = await build_mcp_entity_search_surface(
        manager,
        group_id=group_id,
        name=name,
        entity_type=entity_type,
        limit=limit,
    )
    if result.get("status") != "error":
        _attach_lookup_compat_metadata(
            result,
            tool_name="search_entities",
            recall_companion=recall_companion,
            recall_query=recall_query,
        )
        await recall_middleware(recall_query, result, tool_name="search_entities")
    return result


async def build_api_fact_search_surface(
    manager: Any,
    *,
    group_id: str,
    query: str = "",
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    include_epistemic: bool = False,
    limit: int = 10,
) -> dict:
    """Build the REST fact-search response from the shared lookup facade."""
    results = await _search_facts(
        manager,
        group_id=group_id,
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )
    return {"items": [_api_fact_item(result) for result in results]}


async def build_mcp_fact_search_surface(
    manager: Any,
    *,
    group_id: str,
    query: str = "",
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    include_epistemic: bool = False,
    limit: int = 10,
) -> dict:
    """Build the MCP fact-search response from the shared lookup facade."""
    facts = await _search_facts(
        manager,
        group_id=group_id,
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )
    return {"facts": facts, "total": len(facts)}


async def build_mcp_fact_search_tool_surface(
    manager: Any,
    *,
    group_id: str,
    query: str = "",
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    include_epistemic: bool = False,
    limit: int = 10,
    recall_middleware: Callable[..., Awaitable[None]],
    cfg: Any | None = None,
    project_path: str | None = None,
) -> dict:
    """Build the MCP fact-search compat shim and prime the unified recall path."""
    recall_query = query or subject or predicate or ""
    recall_companion = None
    if cfg is not None:
        recall_companion = await _prime_recall_lookup_path(
            manager,
            group_id=group_id,
            query=recall_query,
            limit=limit,
            cfg=cfg,
            project_path=project_path,
        )
    result = await build_mcp_fact_search_surface(
        manager,
        group_id=group_id,
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )
    _attach_lookup_compat_metadata(
        result,
        tool_name="search_facts",
        recall_companion=recall_companion,
        recall_query=recall_query,
    )
    await recall_middleware(recall_query, result, tool_name="search_facts")
    return result


async def _search_facts(
    manager: Any,
    *,
    group_id: str,
    query: str,
    subject: str | None,
    predicate: str | None,
    include_expired: bool,
    include_epistemic: bool,
    limit: int,
) -> list[dict]:
    return await manager.search_facts(
        group_id=group_id,
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )


def _api_entity_item(result: dict) -> dict:
    return {
        "id": result["id"],
        "name": result["name"],
        "entityType": result["entity_type"],
        "summary": result["summary"],
        "lexicalRegime": result.get("lexical_regime"),
        "canonicalIdentifier": result.get("canonical_identifier"),
        "identifierLabel": bool(result.get("identifier_label", False)),
        "activationCurrent": result["activation_score"],
        "accessCount": result["access_count"],
        "createdAt": result["created_at"],
        "updatedAt": result["updated_at"],
    }


def _api_fact_item(result: dict) -> dict:
    return {
        "subject": result["subject"],
        "predicate": result["predicate"],
        "object": result["object"],
        "validFrom": result.get("valid_from"),
        "validTo": result.get("valid_to"),
        "confidence": result.get("confidence"),
        "sourceEpisode": result.get("source_episode"),
        "createdAt": result.get("created_at"),
    }


class EntityFactLookupService:
    """Own read-only entity and fact lookup for REST/MCP surfaces."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg

    async def resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Resolve an entity ID to its name. Returns ID if not found."""
        entity = await self._graph.get_entity(entity_id, group_id)
        return entity.name if entity else entity_id

    async def search_entities(
        self,
        group_id: str = "default",
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search entities by name and/or type without recording access."""
        now = time.time()
        entities: list[Entity] = []

        if name:
            search_hits = await self._search.search(
                query=name,
                group_id=group_id,
                limit=limit * 2,
            )
            for entity_id, _score in search_hits:
                entity = await self._graph.get_entity(entity_id, group_id)
                if entity and (not entity_type or entity.entity_type == entity_type):
                    entities.append(entity)
                if len(entities) >= limit:
                    break

            if not entities:
                entities = await self._graph.find_entities(
                    name=name,
                    entity_type=entity_type,
                    group_id=group_id,
                    limit=limit,
                )
        else:
            entities = await self._graph.find_entities(
                entity_type=entity_type,
                group_id=group_id,
                limit=limit,
            )

        from engram.extraction.promotion import is_decision_statement_noise

        states = await self._activation.batch_get([entity.id for entity in entities])
        result = []
        for entity in entities:
            # Permanent product hygiene: bootstrap decision_statement scrap is not
            # a first-class Decision for agent lookup surfaces.
            if is_decision_statement_noise(entity.name):
                continue
            state = states.get(entity.id)
            activation_score = 0.0
            access_count = 0
            if state:
                activation_score = compute_activation(state.access_history, now, self._cfg)
                access_count = state.access_count

            result.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "lexical_regime": entity.lexical_regime,
                    "canonical_identifier": entity.canonical_identifier,
                    "identifier_label": entity.identifier_label,
                    "identity_core": bool(getattr(entity, "identity_core", False)),
                    "activation_score": round(activation_score, 4),
                    "created_at": entity.created_at.isoformat() if entity.created_at else None,
                    "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                    "access_count": access_count,
                },
            )
        return result

    async def search_facts(
        self,
        group_id: str = "default",
        query: str = "",
        subject: str | None = None,
        predicate: str | None = None,
        include_expired: bool = False,
        include_epistemic: bool = False,
        limit: int = 10,
    ) -> list[dict]:
        """Search relationships/facts and resolve entity names for the response."""
        normalized_predicate = predicate.upper().replace(" ", "_") if predicate else None
        relationships = await self._find_relationships(
            group_id=group_id,
            query=query,
            subject=subject,
            predicate=normalized_predicate,
            include_expired=include_expired,
            limit=limit,
        )

        result = []
        for relationship in relationships:
            if not include_epistemic and await self.relationship_is_epistemic(
                relationship,
                group_id=group_id,
            ):
                continue
            source_name = await self.resolve_entity_name(relationship.source_id, group_id)
            target_name = await self.resolve_entity_name(relationship.target_id, group_id)
            result.append(
                {
                    "subject": source_name,
                    "predicate": relationship.predicate,
                    "object": target_name,
                    "polarity": relationship.polarity,
                    "valid_from": (
                        relationship.valid_from.isoformat() if relationship.valid_from else None
                    ),
                    "valid_to": (
                        relationship.valid_to.isoformat() if relationship.valid_to else None
                    ),
                    "confidence": relationship.confidence,
                    "source_episode": relationship.source_episode,
                    "created_at": (
                        relationship.created_at.isoformat() if relationship.created_at else None
                    ),
                },
            )
            if len(result) >= limit:
                break
        return result

    async def _find_relationships(
        self,
        *,
        group_id: str,
        query: str,
        subject: str | None,
        predicate: str | None,
        include_expired: bool,
        limit: int,
    ) -> list[Relationship]:
        if subject:
            subject_entities = await self._resolve_subject_entities(
                subject,
                group_id=group_id,
            )
            if not subject_entities:
                return []
            return await self._graph.get_relationships(
                subject_entities[0].id,
                direction="outgoing",
                predicate=predicate,
                active_only=not include_expired,
                group_id=group_id,
            )

        relationships: list[Relationship] = []
        seen_rel_ids: set[str] = set()
        search_hits = await self._search.search(query=query, group_id=group_id, limit=limit)
        for entity_id, _score in search_hits:
            rels = await self._graph.get_relationships(
                entity_id,
                direction="both",
                predicate=predicate,
                active_only=not include_expired,
                group_id=group_id,
            )
            for relationship in rels:
                if relationship.id not in seen_rel_ids:
                    seen_rel_ids.add(relationship.id)
                    relationships.append(relationship)
            if len(relationships) >= limit:
                break
        return relationships

    async def _resolve_subject_entities(self, subject: str, *, group_id: str) -> list[Entity]:
        subject_entities = await self._graph.find_entities(
            name=subject,
            group_id=group_id,
            limit=1,
        )
        if subject_entities:
            return subject_entities

        hits = await self._search.search(query=subject, group_id=group_id, limit=5)
        for entity_id, _score in hits:
            entity = await self._graph.get_entity(entity_id, group_id)
            if entity and entity.name.lower() == subject.lower():
                return [entity]
        return []

    async def relationship_is_epistemic(
        self,
        relationship: Relationship,
        *,
        group_id: str,
    ) -> bool:
        if relationship.predicate in EPISTEMIC_FACT_PREDICATES:
            return True
        source_entity = await self._graph.get_entity(relationship.source_id, group_id)
        target_entity = await self._graph.get_entity(relationship.target_id, group_id)
        source_type = getattr(source_entity, "entity_type", None)
        target_type = getattr(target_entity, "entity_type", None)
        return source_type in {"Decision", "Artifact"} or target_type in {
            "Decision",
            "Artifact",
        }
