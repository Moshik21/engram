"""Graph-state read model for stats, MCP, lifecycle, and dashboard surfaces."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypedDict

from engram.config import ActivationConfig
from engram.storage.protocols import ActivationStore, GraphStore


class TopActivatedEntry(TypedDict):
    id: str
    name: str
    entity_type: str
    summary: str | None
    activation: float
    access_count: int


@dataclass(frozen=True)
class ApiGraphSurface:
    """REST graph payload plus HTTP status."""

    status_code: int
    payload: dict


def graph_entity_not_found_payload(entity_id: str | None) -> dict:
    """Return the REST not-found payload for graph entity lookups."""
    return {"detail": f"Entity '{entity_id}' not found"}


def temporal_graph_invalid_timestamp_payload(value: str) -> dict:
    """Return the REST validation payload for temporal graph timestamps."""
    return {"detail": f"Invalid ISO 8601 timestamp: '{value}'"}


async def build_api_graph_neighborhood_surface(
    manager: Any,
    *,
    group_id: str,
    center: str | None,
    depth: int,
    max_nodes: int,
    min_activation: float,
) -> ApiGraphSurface:
    """Build the REST graph-neighborhood payload through the manager facade."""
    payload = await manager.get_graph_neighborhood(
        group_id=group_id,
        center=center,
        depth=depth,
        max_nodes=max_nodes,
        min_activation=min_activation,
    )
    if payload is None:
        return ApiGraphSurface(
            status_code=404,
            payload=graph_entity_not_found_payload(center),
        )
    return ApiGraphSurface(status_code=200, payload=payload)


async def build_api_temporal_graph_surface(
    manager: Any,
    *,
    group_id: str,
    center: str,
    at: str,
    depth: int,
    max_nodes: int,
) -> ApiGraphSurface:
    """Build the REST temporal graph payload through the manager facade."""
    try:
        at_time = datetime.fromisoformat(at)
    except (ValueError, TypeError):
        return ApiGraphSurface(
            status_code=400,
            payload=temporal_graph_invalid_timestamp_payload(at),
        )

    payload = await manager.get_temporal_graph(
        group_id=group_id,
        center=center,
        at_time=at_time,
        at_label=at,
        depth=depth,
        max_nodes=max_nodes,
    )
    if payload is None:
        return ApiGraphSurface(
            status_code=404,
            payload=graph_entity_not_found_payload(center),
        )
    return ApiGraphSurface(status_code=200, payload=payload)


async def build_mcp_graph_state_surface(
    manager: Any,
    *,
    group_id: str,
    top_n: int = 20,
    include_edges: bool = False,
    entity_types: list[str] | None = None,
) -> dict:
    """Build the MCP graph-state tool payload through the manager facade."""
    return await manager.get_graph_state(
        group_id=group_id,
        top_n=top_n,
        include_edges=include_edges,
        entity_types=entity_types,
    )


async def build_mcp_graph_stats_resource_surface(
    manager: Any,
    *,
    group_id: str,
) -> dict:
    """Build the MCP graph stats resource payload."""
    state = await build_mcp_graph_state_surface(
        manager,
        group_id=group_id,
        top_n=10,
        include_edges=False,
    )
    return state["stats"]


async def build_mcp_entity_profile_resource_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
) -> dict:
    """Build the MCP entity profile resource payload through the manager facade."""
    return await manager.get_entity_profile(entity_id, group_id)


async def build_mcp_entity_neighbors_resource_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
) -> dict:
    """Build the MCP entity neighbors resource payload through the manager facade."""
    return await manager.get_entity_neighbors(entity_id, group_id)


class GraphStateService:
    """Build the shared graph-state read model exposed through GraphManager."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cfg: ActivationConfig,
        get_recall_metrics: Callable[[str], dict],
        get_epistemic_metrics: Callable[[str], dict],
        resolve_entity_name: Callable[[str, str], Awaitable[str]],
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._get_recall_metrics = get_recall_metrics
        self._get_epistemic_metrics = get_epistemic_metrics
        self._resolve_entity_name = resolve_entity_name

    def get_graph_store(self) -> GraphStore:
        """Return the graph store used by lifecycle summary read models."""
        return self._graph

    async def list_episode_summaries(
        self,
        *,
        group_id: str = "default",
        cursor: str | None = None,
        limit: int = 50,
        source: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Return the REST episode dashboard listing payload."""
        episodes, next_cursor = await self._graph.get_episodes_paginated(
            group_id=group_id,
            cursor=cursor,
            limit=limit,
            source=source,
            status=status,
        )

        items = []
        get_episode_cue = getattr(self._graph, "get_episode_cue", None)
        for episode in episodes:
            cue = await get_episode_cue(episode.id, group_id) if get_episode_cue else None
            items.append(self._build_episode_summary_item(episode, cue))

        return {
            "items": items,
            "nextCursor": next_cursor,
            "total": len(items),
        }

    async def get_dashboard_stats(
        self,
        *,
        group_id: str = "default",
        days: int = 30,
    ) -> dict:
        """Return the dashboard stats overview payload."""
        result = await self.get_graph_state(
            group_id=group_id,
            top_n=20,
            include_edges=False,
        )
        top_activated = [
            {
                "id": item["id"],
                "name": item["name"],
                "entityType": item["entity_type"],
                "summary": item["summary"],
                "activationCurrent": item["activation"],
                "accessCount": item["access_count"],
            }
            for item in result.get("top_activated", [])
        ]
        top_connected = await self._graph.get_top_connected(group_id=group_id, limit=10)
        growth_timeline = await self._graph.get_growth_timeline(group_id=group_id, days=days)
        return {
            "stats": result["stats"],
            "topActivated": top_activated,
            "topConnected": top_connected,
            "growthTimeline": growth_timeline,
            "groupId": result["group_id"],
        }

    async def get_activation_snapshot(
        self,
        *,
        group_id: str = "default",
        limit: int = 50,
    ) -> dict:
        """Return top activated entities for the dashboard activation monitor."""
        from engram.activation.engine import compute_activation

        now = time.time()
        top = await self._activation.get_top_activated(group_id=group_id, limit=limit * 2)
        items = []
        for entity_id, state in top:
            entity = await self._graph.get_entity(entity_id, group_id)
            if not entity:
                continue

            activation = compute_activation(state.access_history, now, self._cfg)
            items.append(
                {
                    "entityId": entity.id,
                    "name": entity.name,
                    "entityType": entity.entity_type,
                    "currentActivation": round(activation, 4),
                    "accessCount": state.access_count,
                    "lastAccessedAt": (
                        datetime.fromtimestamp(state.last_accessed).isoformat()
                        if state.last_accessed
                        else None
                    ),
                    "decayRate": self._cfg.decay_exponent,
                }
            )
            if len(items) >= limit:
                break

        items.sort(key=lambda item: item["currentActivation"], reverse=True)
        return {"topActivated": items}

    async def get_activation_curve(
        self,
        *,
        group_id: str = "default",
        entity_id: str,
        hours: int = 24,
        points: int = 48,
    ) -> dict | None:
        """Return an ACT-R decay curve for one entity."""
        from engram.activation.engine import compute_activation

        entity = await self._graph.get_entity(entity_id, group_id)
        if not entity:
            return None

        state = await self._activation.get_activation(entity_id)
        access_history = state.access_history if state else []

        now = time.time()
        start_time = now - (hours * 3600)
        step = (now - start_time) / max(points - 1, 1)

        curve: list[dict] = []
        for index in range(points):
            timestamp = start_time + (index * step)
            history_at_timestamp = [ts for ts in access_history if ts <= timestamp]
            activation = (
                compute_activation(history_at_timestamp, timestamp, self._cfg)
                if history_at_timestamp
                else 0.0
            )
            curve.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "activation": round(activation, 4),
                }
            )

        access_events = [
            datetime.fromtimestamp(ts).isoformat()
            for ts in access_history
            if start_time <= ts <= now
        ]
        formula = f"B_i = ln(Σ t_j^{{-{self._cfg.decay_exponent}}})"

        return {
            "entityId": entity_id,
            "entityName": entity.name,
            "curve": curve,
            "accessEvents": access_events,
            "formula": formula,
            "hours": hours,
            "points": points,
        }

    async def get_graph_neighborhood(
        self,
        *,
        group_id: str = "default",
        center: str | None = None,
        depth: int = 2,
        max_nodes: int = 2000,
        min_activation: float = 0.0,
    ) -> dict | None:
        """Return the dashboard graph-neighborhood payload."""
        now = time.time()

        if not center:
            all_entities = await self._graph.find_entities(group_id=group_id, limit=max_nodes)
            if not all_entities:
                return {
                    "centerId": None,
                    "nodes": [],
                    "edges": [],
                    "representation": self._build_graph_representation(
                        scope="neighborhood",
                        layout="force",
                        represented_entity_count=0,
                        represented_edge_count=0,
                        displayed_node_count=0,
                        displayed_edge_count=0,
                        truncated=False,
                    ),
                    "truncated": False,
                    "totalInNeighborhood": 0,
                }

            entity_ids = {entity.id for entity in all_entities}
            states = await self._activation.batch_get(list(entity_ids))

            nodes: list[dict[str, Any]] = []
            for entity in all_entities:
                node = self._build_graph_node(entity, states.get(entity.id), now)
                if min_activation > 0.0 and node["activationCurrent"] < min_activation:
                    continue
                nodes.append(node)

            best = max(nodes, key=lambda node: node["activationCurrent"]) if nodes else None
            resolved_center = best["id"] if best else None

            total = len(nodes)
            truncated = False
            if len(nodes) > max_nodes:
                nodes.sort(key=lambda node: node["activationCurrent"], reverse=True)
                nodes = nodes[:max_nodes]
                truncated = True

            remaining_ids = {node["id"] for node in nodes}
            all_rels = await self._graph.get_all_edges(
                group_id=group_id,
                entity_ids=remaining_ids,
                limit=max_nodes * 5,
            )
            edges = [self._build_graph_edge(relationship) for relationship in all_rels]

            return {
                "centerId": resolved_center,
                "nodes": nodes,
                "edges": edges,
                "representation": self._build_graph_representation(
                    scope="neighborhood",
                    layout="force",
                    represented_entity_count=total,
                    represented_edge_count=len(all_rels),
                    displayed_node_count=len(nodes),
                    displayed_edge_count=len(edges),
                    truncated=truncated,
                ),
                "truncated": truncated,
                "totalInNeighborhood": total,
            }

        center_entity = await self._graph.get_entity(center, group_id)
        if not center_entity:
            return None

        neighbor_pairs = await self._graph.get_neighbors(
            center,
            hops=depth,
            group_id=group_id,
            max_results=max_nodes * 3,
        )

        entities_map: dict[str, Any] = {center: center_entity}
        edges_map: dict[str, Any] = {}

        for entity, relationship in neighbor_pairs:
            entities_map[entity.id] = entity
            edges_map[relationship.id] = relationship

        neighborhood_entity_ids = list(entities_map)
        states = await self._activation.batch_get(neighborhood_entity_ids)

        neighborhood_nodes: list[dict[str, Any]] = []
        for entity_id, entity in entities_map.items():
            neighborhood_nodes.append(self._build_graph_node(entity, states.get(entity_id), now))

        total_in_neighborhood = len(neighborhood_nodes)

        if min_activation > 0.0:
            neighborhood_nodes = [
                node for node in neighborhood_nodes if node["activationCurrent"] >= min_activation
            ]

        truncated = False
        if len(neighborhood_nodes) > max_nodes:
            neighborhood_nodes.sort(key=lambda node: node["activationCurrent"], reverse=True)
            neighborhood_nodes = neighborhood_nodes[:max_nodes]
            truncated = True

        remaining_ids = {node["id"] for node in neighborhood_nodes}
        edges = [
            self._build_graph_edge(relationship)
            for relationship in edges_map.values()
            if relationship.source_id in remaining_ids and relationship.target_id in remaining_ids
        ]

        return {
            "centerId": center,
            "nodes": neighborhood_nodes,
            "edges": edges,
            "representation": self._build_graph_representation(
                scope="neighborhood",
                layout="force",
                represented_entity_count=total_in_neighborhood,
                represented_edge_count=len(edges_map),
                displayed_node_count=len(neighborhood_nodes),
                displayed_edge_count=len(edges),
                truncated=truncated,
            ),
            "truncated": truncated,
            "totalInNeighborhood": total_in_neighborhood,
        }

    async def get_temporal_graph(
        self,
        *,
        group_id: str = "default",
        center: str,
        at_time: datetime,
        at_label: str,
        depth: int = 2,
        max_nodes: int = 2000,
    ) -> dict | None:
        """Return the dashboard temporal graph payload for a point in time."""
        center_entity = await self._graph.get_entity(center, group_id)
        if not center_entity:
            return None

        visited: set[str] = {center}
        frontier: set[str] = {center}
        entities_map: dict[str, Any] = {center: center_entity}
        edges_list: list[dict[str, Any]] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for entity_id in frontier:
                relationships = await self._graph.get_relationships_at(
                    entity_id,
                    at_time,
                    group_id=group_id,
                )
                for relationship in relationships:
                    other_id = (
                        relationship.target_id
                        if relationship.source_id == entity_id
                        else relationship.source_id
                    )
                    edges_list.append(self._build_graph_edge(relationship))
                    if other_id not in visited:
                        visited.add(other_id)
                        next_frontier.add(other_id)
            if next_frontier:
                batch = await self._graph.batch_get_entities(list(next_frontier), group_id)
                entities_map.update(batch)
            frontier = next_frontier
            if not frontier:
                break

        nodes = [
            self._build_temporal_graph_node(entity)
            for _entity_id, entity in list(entities_map.items())[:max_nodes]
        ]

        remaining_ids = {node["id"] for node in nodes}
        seen_edge_ids: set[str] = set()
        edges: list[dict[str, Any]] = []
        for edge in edges_list:
            in_scope = edge["source"] in remaining_ids and edge["target"] in remaining_ids
            if edge["id"] not in seen_edge_ids and in_scope:
                seen_edge_ids.add(edge["id"])
                edges.append(edge)

        truncated = len(entities_map) > max_nodes
        return {
            "centerId": center,
            "at": at_label,
            "nodes": nodes,
            "edges": edges,
            "representation": self._build_graph_representation(
                scope="temporal",
                layout="force",
                represented_entity_count=len(entities_map),
                represented_edge_count=len(edges_list),
                displayed_node_count=len(nodes),
                displayed_edge_count=len(edges),
                truncated=truncated,
            ),
            "truncated": truncated,
            "totalInNeighborhood": len(entities_map),
        }

    async def get_graph_state(
        self,
        group_id: str = "default",
        top_n: int = 20,
        include_edges: bool = False,
        entity_types: list[str] | None = None,
    ) -> dict:
        """Return graph statistics and top-activated nodes."""
        from engram.activation.engine import compute_activation

        now = time.time()

        stats = await self._graph.get_stats(group_id)
        type_counts = await self._graph.get_entity_type_counts(group_id)
        stats["entity_type_distribution"] = type_counts

        top = await self._activation.get_top_activated(group_id=group_id, limit=top_n * 2)

        top_activated: list[TopActivatedEntry] = []
        active_count = 0
        dormant_count = 0

        for entity_id, state in top:
            entity = await self._graph.get_entity(entity_id, group_id)
            if not entity:
                continue
            if entity_types and entity.entity_type not in entity_types:
                continue

            activation = compute_activation(state.access_history, now, self._cfg)
            if activation > 0.3:
                active_count += 1
            else:
                dormant_count += 1

            if len(top_activated) < top_n:
                top_activated.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "summary": entity.summary,
                        "activation": round(activation, 4),
                        "access_count": state.access_count,
                    }
                )

        stats["active_entities"] = active_count
        stats["dormant_entities"] = dormant_count
        stats["recall_metrics"] = self._get_recall_metrics(group_id)
        stats["epistemic_metrics"] = self._get_epistemic_metrics(group_id)

        result: dict = {
            "stats": stats,
            "top_activated": top_activated,
            "group_id": group_id,
        }

        if include_edges:
            result["edges"] = await self._build_edges(
                top_activated,
                group_id=group_id,
            )

        return result

    async def get_entity_profile(self, entity_id: str, group_id: str = "default") -> dict:
        """Return a single entity profile for MCP/resource clients."""
        from engram.activation.engine import compute_activation

        entity = await self._graph.get_entity(entity_id, group_id)
        if not entity:
            return {"error": "Entity not found", "entity_id": entity_id}

        state = await self._activation.get_activation(entity_id)
        relationships = await self._graph.get_relationships(
            entity_id,
            active_only=True,
            group_id=group_id,
        )

        facts = []
        for relationship in relationships:
            target_name = await self._resolve_entity_name(relationship.target_id, group_id)
            source_name = await self._resolve_entity_name(relationship.source_id, group_id)
            if relationship.source_id == entity_id:
                facts.append(
                    {
                        "predicate": relationship.predicate,
                        "object": target_name,
                        "valid_from": (
                            relationship.valid_from.isoformat() if relationship.valid_from else None
                        ),
                        "valid_to": (
                            relationship.valid_to.isoformat() if relationship.valid_to else None
                        ),
                    }
                )
            else:
                facts.append(
                    {
                        "predicate": relationship.predicate,
                        "subject": source_name,
                        "valid_from": (
                            relationship.valid_from.isoformat() if relationship.valid_from else None
                        ),
                        "valid_to": (
                            relationship.valid_to.isoformat() if relationship.valid_to else None
                        ),
                    }
                )

        now = time.time()
        activation_score = 0.0
        if state:
            activation_score = compute_activation(state.access_history, now, self._cfg)

        return {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "summary": entity.summary,
            "activation": {
                "current": round(activation_score, 4),
                "access_count": state.access_count if state else 0,
                "last_accessed": state.last_accessed if state else None,
            },
            "facts": facts,
            "created_at": entity.created_at.isoformat() if entity.created_at else None,
            "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
        }

    async def get_entity_detail(self, entity_id: str, group_id: str = "default") -> dict | None:
        """Return the REST entity detail view with facts and activation."""
        from engram.activation.engine import compute_activation

        entity = await self._graph.get_entity(entity_id, group_id)
        if not entity:
            return None

        state = await self._activation.get_activation(entity_id)
        activation_current = 0.0
        access_count = 0
        last_accessed = None
        if state and state.access_history:
            activation_current = compute_activation(state.access_history, time.time(), self._cfg)
            access_count = state.access_count
            if state.last_accessed:
                last_accessed = datetime.fromtimestamp(
                    state.last_accessed,
                    tz=timezone.utc,
                ).isoformat()
        else:
            activation_current = getattr(entity, "activation_current", 0.0) or 0.0
            access_count = getattr(entity, "access_count", 0) or 0
            entity_last_accessed = getattr(entity, "last_accessed", None)
            if entity_last_accessed:
                last_accessed = (
                    entity_last_accessed.isoformat()
                    if hasattr(entity_last_accessed, "isoformat")
                    else str(entity_last_accessed)
                )

        relationships = await self._graph.get_relationships(
            entity_id,
            active_only=True,
            group_id=group_id,
        )
        other_ids = list(
            {
                relationship.target_id
                if relationship.source_id == entity_id
                else relationship.source_id
                for relationship in relationships
            }
        )
        related_entities = (
            await self._graph.batch_get_entities(other_ids, group_id) if other_ids else {}
        )

        facts = []
        for relationship in relationships:
            if relationship.source_id == entity_id:
                direction = "outgoing"
                other_id = relationship.target_id
            else:
                direction = "incoming"
                other_id = relationship.source_id

            other_entity = related_entities.get(other_id)
            other_info = (
                {
                    "id": other_entity.id,
                    "name": other_entity.name,
                    "entityType": other_entity.entity_type,
                }
                if other_entity
                else {"id": other_id, "name": other_id, "entityType": "Unknown"}
            )
            facts.append(
                {
                    "id": relationship.id,
                    "predicate": relationship.predicate,
                    "direction": direction,
                    "other": other_info,
                    "weight": relationship.weight,
                    "validFrom": (
                        relationship.valid_from.isoformat() if relationship.valid_from else None
                    ),
                    "validTo": (
                        relationship.valid_to.isoformat() if relationship.valid_to else None
                    ),
                    "createdAt": (
                        relationship.created_at.isoformat() if relationship.created_at else None
                    ),
                }
            )

        return {
            "id": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "summary": entity.summary,
            "lexicalRegime": entity.lexical_regime,
            "canonicalIdentifier": entity.canonical_identifier,
            "identifierLabel": entity.identifier_label,
            "activationCurrent": round(activation_current, 4),
            "accessCount": access_count,
            "lastAccessed": last_accessed,
            "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
            "facts": facts,
        }

    async def get_entity_neighbors(self, entity_id: str, group_id: str = "default") -> list[dict]:
        """Return one-hop entity neighbors for MCP/resource clients."""
        neighbors = await self._graph.get_neighbors(entity_id, hops=1, group_id=group_id)
        result = []
        for entity, relationship in neighbors:
            result.append(
                {
                    "entity": {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "summary": entity.summary,
                    },
                    "relationship": {
                        "predicate": relationship.predicate,
                        "source_id": relationship.source_id,
                        "target_id": relationship.target_id,
                        "weight": relationship.weight,
                    },
                }
            )
        return result

    async def _build_edges(
        self,
        top_activated: list[TopActivatedEntry],
        *,
        group_id: str,
    ) -> list[dict]:
        edges = []
        seen_rel_ids: set[str] = set()
        for entry in top_activated:
            relationships = await self._graph.get_relationships(
                entry["id"],
                active_only=True,
                group_id=group_id,
            )
            for relationship in relationships:
                if relationship.id in seen_rel_ids:
                    continue
                seen_rel_ids.add(relationship.id)
                source_name = await self._resolve_entity_name(relationship.source_id, group_id)
                target_name = await self._resolve_entity_name(relationship.target_id, group_id)
                edges.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "predicate": relationship.predicate,
                        "weight": relationship.weight,
                    }
                )
        return edges

    def _build_graph_node(self, entity: Any, state: Any, now: float) -> dict[str, Any]:
        """Build a dashboard graph node from an entity and activation state."""
        from engram.activation.engine import compute_activation

        activation_current = 0.0
        access_count = 0
        last_accessed = None
        if state and state.access_history:
            activation_current = compute_activation(state.access_history, now, self._cfg)
            access_count = state.access_count
            if state.last_accessed:
                last_accessed = datetime.fromtimestamp(
                    state.last_accessed,
                    tz=timezone.utc,
                ).isoformat()
        else:
            activation_current = getattr(entity, "activation_current", 0.0) or 0.0
            access_count = getattr(entity, "access_count", 0) or 0
            entity_last_accessed = getattr(entity, "last_accessed", None)
            if entity_last_accessed:
                last_accessed = (
                    entity_last_accessed.isoformat()
                    if hasattr(entity_last_accessed, "isoformat")
                    else str(entity_last_accessed)
                )

        return {
            "id": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "summary": entity.summary,
            "activationCurrent": round(activation_current, 4),
            "accessCount": access_count,
            "lastAccessed": last_accessed,
            "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
        }

    def _build_temporal_graph_node(self, entity: Any) -> dict[str, Any]:
        """Build a temporal graph node without applying current activation."""
        return {
            "id": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "summary": entity.summary,
            "activationCurrent": 0.0,
            "accessCount": 0,
            "lastAccessed": None,
            "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
        }

    def _build_episode_summary_item(self, episode: Any, cue: Any | None) -> dict:
        return {
            "episodeId": episode.id,
            "content": episode.content[:200] if episode.content else None,
            "source": episode.source,
            "status": self._enum_value(episode.status),
            "projectionState": self._enum_value(getattr(episode, "projection_state", None)),
            "lastProjectionReason": getattr(episode, "last_projection_reason", None),
            "lastProjectedAt": self._iso_z(getattr(episode, "last_projected_at", None)),
            "conversationDate": self._iso_z(getattr(episode, "conversation_date", None)),
            "createdAt": self._iso_z(episode.created_at),
            "updatedAt": self._iso_z(episode.updated_at),
            "error": episode.error,
            "retryCount": episode.retry_count,
            "processingDurationMs": episode.processing_duration_ms,
            "entities": [],
            "factsCount": 0,
            "cue": self._build_episode_cue_summary(cue),
        }

    def _build_episode_cue_summary(self, cue: Any | None) -> dict | None:
        if cue is None:
            return None
        return {
            "cueText": cue.cue_text[:240] if cue.cue_text else None,
            "projectionState": self._enum_value(cue.projection_state),
            "routeReason": cue.route_reason,
            "hitCount": cue.hit_count,
            "surfacedCount": cue.surfaced_count,
            "selectedCount": cue.selected_count,
            "usedCount": cue.used_count,
            "nearMissCount": cue.near_miss_count,
            "policyScore": cue.policy_score,
            "projectionAttempts": cue.projection_attempts,
            "lastHitAt": self._iso_z(cue.last_hit_at),
            "lastFeedbackAt": self._iso_z(cue.last_feedback_at),
            "lastProjectedAt": self._iso_z(cue.last_projected_at),
        }

    def _build_graph_edge(self, relationship: Any) -> dict[str, Any]:
        """Build a dashboard graph edge from a relationship."""
        return {
            "id": relationship.id,
            "source": relationship.source_id,
            "target": relationship.target_id,
            "predicate": relationship.predicate,
            "weight": relationship.weight,
            "validFrom": (relationship.valid_from.isoformat() if relationship.valid_from else None),
            "validTo": relationship.valid_to.isoformat() if relationship.valid_to else None,
            "createdAt": (relationship.created_at.isoformat() if relationship.created_at else None),
        }

    @staticmethod
    def _enum_value(value: object) -> str | None:
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            return enum_value
        return value if isinstance(value, str) else None

    @staticmethod
    def _iso_z(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return f"{value.isoformat()}Z"

    def _build_graph_representation(
        self,
        *,
        scope: str,
        layout: str,
        represented_entity_count: int,
        represented_edge_count: int,
        displayed_node_count: int,
        displayed_edge_count: int,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "scope": scope,
            "layout": layout,
            "representedEntityCount": represented_entity_count,
            "representedEdgeCount": represented_edge_count,
            "displayedNodeCount": displayed_node_count,
            "displayedEdgeCount": displayed_edge_count,
            "truncated": truncated,
        }
