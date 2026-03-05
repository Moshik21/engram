"""FalkorDB implementation of GraphStore protocol (Full mode)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta

from engram.config import FalkorDBConfig
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.protocols import ENTITY_UPDATABLE_FIELDS, EPISODE_UPDATABLE_FIELDS
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


class FalkorDBGraphStore:
    """Graph store backed by FalkorDB (Redis-based graph database)."""

    def __init__(self, config: FalkorDBConfig, encryptor=None) -> None:
        self._config = config
        self._encryptor = encryptor
        self._db = None
        self._graph = None

    async def _query(self, cypher: str, params: dict | None = None, timeout: int | None = None):
        """Execute a Cypher query via thread pool (falkordb is synchronous)."""
        kwargs: dict = {"params": params or {}}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await asyncio.to_thread(self._graph.query, cypher, **kwargs)

    async def initialize(self) -> None:
        """Connect to FalkorDB and create indexes."""
        from falkordb import FalkorDB

        password = self._config.password or None
        kwargs: dict = {
            "host": self._config.host,
            "port": self._config.port,
            "password": password,
        }
        if self._config.ssl:
            kwargs["ssl"] = True
            if self._config.ssl_ca_cert:
                kwargs["ssl_ca_certs"] = self._config.ssl_ca_cert
        self._db = await asyncio.to_thread(FalkorDB, **kwargs)
        self._graph = await asyncio.to_thread(self._db.select_graph, self._config.graph_name)

        # Create indexes — FalkorDB errors on duplicate, so wrap each in try/except
        indexes = [
            "CREATE INDEX FOR (n:Entity) ON (n.id)",
            "CREATE INDEX FOR (n:Entity) ON (n.group_id)",
            "CREATE INDEX FOR (n:Entity) ON (n.name)",
            "CREATE INDEX FOR (n:Entity) ON (n.entity_type)",
            "CREATE INDEX FOR (n:Entity) ON (n.deleted_at)",
            "CREATE INDEX FOR (n:Episode) ON (n.id)",
            "CREATE INDEX FOR (n:Episode) ON (n.group_id)",
            "CREATE INDEX FOR (n:Episode) ON (n.status)",
            "CREATE INDEX FOR (n:Episode) ON (n.created_at)",
            "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.predicate)",
            "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_to)",
            "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.group_id)",
            "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.source_id)",
            "CREATE INDEX FOR (n:Intention) ON (n.id)",
            "CREATE INDEX FOR (n:Intention) ON (n.group_id)",
        ]
        for idx in indexes:
            try:
                await self._query(idx)
            except Exception:
                pass  # Index already exists

        logger.info(
            "FalkorDB graph store initialized (host=%s, port=%d, graph=%s)",
            self._config.host,
            self._config.port,
            self._config.graph_name,
        )

    async def close(self) -> None:
        """No-op — falkordb sync client has no async close."""
        pass

    # --- Encryption helpers ---

    def _encrypt(self, group_id: str, plaintext: str | None) -> str | None:
        if not plaintext or not self._encryptor:
            return plaintext
        return self._encryptor.encrypt(group_id, plaintext)

    def _decrypt(self, group_id: str, data: str | None) -> str | None:
        if not data or not self._encryptor:
            return data
        return self._encryptor.decrypt(group_id, data)

    # --- Entities ---

    async def create_entity(self, entity: Entity) -> str:
        now = datetime.utcnow().isoformat()
        summary = self._encrypt(entity.group_id, entity.summary)
        await self._query(
            """CREATE (n:Entity {
                id: $id, name: $name, entity_type: $entity_type,
                summary: $summary, attributes: $attributes,
                group_id: $group_id,
                created_at: $created_at, updated_at: $updated_at,
                activation_current: $activation_current,
                access_count: $access_count, last_accessed: $last_accessed,
                pii_detected: $pii_detected, pii_categories: $pii_categories
            })""",
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "summary": summary,
                "attributes": json.dumps(entity.attributes) if entity.attributes else None,
                "group_id": entity.group_id,
                "created_at": entity.created_at.isoformat() if entity.created_at else now,
                "updated_at": now,
                "activation_current": entity.activation_current,
                "access_count": entity.access_count,
                "last_accessed": (
                    entity.last_accessed.isoformat() if entity.last_accessed else None
                ),
                "pii_detected": entity.pii_detected,
                "pii_categories": (
                    json.dumps(entity.pii_categories) if entity.pii_categories else None
                ),
            },
        )
        return entity.id

    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None:
        result = await self._query(
            """MATCH (n:Entity {id: $id, group_id: $gid})
               WHERE n.deleted_at IS NULL
               RETURN n""",
            {"id": entity_id, "gid": group_id},
        )
        if not result.result_set:
            return None
        node = result.result_set[0][0]
        return self._node_to_entity(node, group_id)

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        if not updates:
            return
        updates["updated_at"] = datetime.utcnow().isoformat()
        invalid = set(updates.keys()) - ENTITY_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed entity update fields: {invalid}")
        set_parts = []
        for key in updates:
            set_parts.append(f"n.{key} = ${key}")
        set_clause = ", ".join(set_parts)
        params = dict(updates)
        params["id"] = entity_id
        params["gid"] = group_id
        await self._query(
            f"MATCH (n:Entity {{id: $id, group_id: $gid}}) SET {set_clause}",
            params,
        )

    async def delete_entity(self, entity_id: str, soft: bool = True, *, group_id: str) -> None:
        if soft:
            await self._query(
                "MATCH (n:Entity {id: $id, group_id: $gid}) SET n.deleted_at = $deleted_at",
                {"id": entity_id, "gid": group_id, "deleted_at": datetime.utcnow().isoformat()},
            )
        else:
            await self._query(
                "MATCH (n:Entity {id: $id, group_id: $gid}) DETACH DELETE n",
                {"id": entity_id, "gid": group_id},
            )

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        conditions = ["n.deleted_at IS NULL"]
        params: dict = {"limit": limit}
        if name:
            conditions.append("toLower(n.name) CONTAINS toLower($name)")
            params["name"] = name
        if entity_type:
            conditions.append("n.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if group_id:
            conditions.append("n.group_id = $group_id")
            params["group_id"] = group_id
        where = " AND ".join(conditions)
        result = await self._query(
            f"MATCH (n:Entity) WHERE {where} RETURN n ORDER BY n.updated_at DESC LIMIT $limit",
            params,
        )
        return [self._node_to_entity(row[0], group_id) for row in result.result_set]

    async def find_entity_candidates(
        self, name: str, group_id: str, limit: int = 30,
    ) -> list[Entity]:
        """Retrieve candidate entities for fuzzy resolution via CONTAINS search."""
        seen_ids: set[str] = set()
        results: list[Entity] = []

        # Phase 1: Full name CONTAINS match (indexed on n.name)
        result = await self._query(
            """MATCH (n:Entity)
               WHERE toLower(n.name) CONTAINS toLower($name)
                 AND n.group_id = $gid AND n.deleted_at IS NULL
               RETURN n LIMIT $limit""",
            {"name": name, "gid": group_id, "limit": limit},
        )
        for row in result.result_set:
            entity = self._node_to_entity(row[0], group_id)
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 2: Token fallback — search individual tokens >= 3 chars
        tokens = [t for t in name.strip().split() if len(t) >= 3]
        for token in tokens:
            if len(results) >= limit:
                break
            token_result = await self._query(
                """MATCH (n:Entity)
                   WHERE toLower(n.name) CONTAINS toLower($token)
                     AND n.group_id = $gid AND n.deleted_at IS NULL
                   RETURN n LIMIT $remaining""",
                {"token": token, "gid": group_id, "remaining": limit - len(results)},
            )
            for row in token_result.result_set:
                entity = self._node_to_entity(row[0], group_id)
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    results.append(entity)

        return results[:limit]

    # --- Relationships ---

    async def create_relationship(self, rel: Relationship) -> str:
        await self._query(
            """MATCH (s:Entity {id: $src}), (t:Entity {id: $tgt})
               CREATE (s)-[r:RELATES_TO {
                   id: $id, source_id: $src, target_id: $tgt,
                   predicate: $predicate, weight: $weight,
                   valid_from: $valid_from, valid_to: $valid_to,
                   created_at: $created_at, confidence: $confidence,
                   source_episode: $source_episode, group_id: $group_id
               }]->(t)""",
            {
                "id": rel.id,
                "src": rel.source_id,
                "tgt": rel.target_id,
                "predicate": rel.predicate,
                "weight": rel.weight,
                "valid_from": rel.valid_from.isoformat() if rel.valid_from else None,
                "valid_to": rel.valid_to.isoformat() if rel.valid_to else None,
                "created_at": (
                    rel.created_at.isoformat() if rel.created_at else datetime.utcnow().isoformat()
                ),
                "confidence": rel.confidence,
                "source_episode": rel.source_episode,
                "group_id": rel.group_id,
            },
        )
        return rel.id

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: str | None = None,
        active_only: bool = True,
        group_id: str = "default",
    ) -> list[Relationship]:
        if direction == "outgoing":
            match = "MATCH (s:Entity {id: $eid})-[r:RELATES_TO]->(t:Entity)"
        elif direction == "incoming":
            match = "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity {id: $eid})"
        else:
            match = "MATCH (s:Entity)-[r:RELATES_TO]-(t:Entity) WHERE s.id = $eid OR t.id = $eid"

        conditions = []
        params: dict = {"eid": entity_id, "now": datetime.utcnow().isoformat()}
        if predicate:
            conditions.append("r.predicate = $predicate")
            params["predicate"] = predicate
        if active_only:
            conditions.append("(r.valid_to IS NULL OR r.valid_to > $now)")
        conditions.append("r.group_id = $gid")
        params["gid"] = group_id

        where = ""
        if conditions:
            if direction == "both":
                where = " AND " + " AND ".join(conditions)
            else:
                where = " WHERE " + " AND ".join(conditions)

        cypher = f"{match}{where} RETURN r"
        result = await self._query(cypher, params)

        seen_ids: set[str] = set()
        rels = []
        for row in result.result_set:
            rel = self._edge_to_relationship(row[0])
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                rels.append(rel)
        return rels

    async def invalidate_relationship(self, rel_id: str, valid_to: datetime, group_id: str) -> None:
        await self._query(
            """MATCH ()-[r:RELATES_TO {id: $id}]->()
               WHERE r.group_id = $gid
               SET r.valid_to = $valid_to""",
            {"id": rel_id, "gid": group_id, "valid_to": valid_to.isoformat()},
        )

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        weight_delta: float,
        max_weight: float = 3.0,
        group_id: str = "default",
    ) -> float | None:
        """Atomically increment edge weight in FalkorDB, capped at max_weight."""
        result = await self._query(
            """MATCH (s:Entity)-[r:RELATES_TO]-(t:Entity)
               WHERE s.id = $src AND t.id = $tgt
                 AND r.group_id = $gid
                 AND (r.valid_to IS NULL OR r.valid_to > $now)
               SET r.weight = CASE
                   WHEN r.weight + $delta > $max_w THEN $max_w
                   ELSE r.weight + $delta
               END
               RETURN r.weight""",
            {
                "src": source_id,
                "tgt": target_id,
                "delta": weight_delta,
                "max_w": max_weight,
                "gid": group_id,
                "now": datetime.utcnow().isoformat(),
            },
        )
        if result.result_set:
            return result.result_set[0][0]
        return None

    async def find_conflicting_relationships(
        self,
        source_id: str,
        predicate: str,
        group_id: str,
    ) -> list[Relationship]:
        result = await self._query(
            """MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity)
               WHERE r.predicate = $predicate
                 AND r.group_id = $group_id
                 AND r.valid_to IS NULL
               RETURN r""",
            {"src": source_id, "predicate": predicate, "group_id": group_id},
        )
        return [self._edge_to_relationship(row[0]) for row in result.result_set]

    async def find_existing_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        group_id: str,
    ) -> Relationship | None:
        """Find an active relationship matching (source, target, predicate)."""
        now = datetime.utcnow().isoformat()
        result = await self._query(
            """MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt})
               WHERE r.predicate = $predicate
                 AND r.group_id = $gid
                 AND (r.valid_to IS NULL OR r.valid_to > $now)
               RETURN r LIMIT 1""",
            {"src": source_id, "tgt": target_id, "predicate": predicate,
             "gid": group_id, "now": now},
        )
        if result.result_set:
            return self._edge_to_relationship(result.result_set[0][0])
        return None

    async def get_relationships_at(
        self,
        entity_id: str,
        at_time: datetime,
        direction: str = "both",
        group_id: str = "default",
    ) -> list[Relationship]:
        t = at_time.isoformat()
        if direction == "outgoing":
            match = "MATCH (s:Entity {id: $eid})-[r:RELATES_TO]->(t:Entity)"
        elif direction == "incoming":
            match = "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity {id: $eid})"
        else:
            match = "MATCH (s:Entity)-[r:RELATES_TO]-(t:Entity) WHERE s.id = $eid OR t.id = $eid"

        time_conditions = [
            "(r.valid_from IS NULL OR r.valid_from <= $at_time)",
            "(r.valid_to IS NULL OR r.valid_to > $at_time)",
        ]
        params: dict = {"eid": entity_id, "at_time": t}
        time_conditions.append("r.group_id = $gid")
        params["gid"] = group_id
        time_filter = " AND ".join(time_conditions)
        if direction == "both":
            where = f" AND {time_filter}"
        else:
            where = f" WHERE {time_filter}"

        result = await self._query(
            f"{match}{where} RETURN r",
            params,
        )

        seen_ids: set[str] = set()
        rels = []
        for row in result.result_set:
            rel = self._edge_to_relationship(row[0])
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                rels.append(rel)
        return rels

    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        group_id: str | None = None,
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops via iterative BFS (avoids combinatorial explosion)."""
        now_str = datetime.utcnow().isoformat()
        group_filter = " AND m.group_id = $group_id" if group_id else ""

        seen_entities: set[str] = {entity_id}
        seen_edges: set[str] = set()
        frontier: set[str] = {entity_id}
        results: list[tuple[Entity, Relationship]] = []

        for _hop in range(hops):
            if not frontier:
                break

            # Fetch 1-hop neighbors from current frontier
            params: dict = {"ids": list(frontier), "now": now_str}
            if group_id:
                params["group_id"] = group_id

            result = await self._query(
                f"""MATCH (c:Entity)-[edge:RELATES_TO]-(m:Entity)
                    WHERE c.id IN $ids
                      AND m.deleted_at IS NULL
                      AND (edge.valid_to IS NULL OR edge.valid_to > $now)
                      {group_filter}
                    RETURN DISTINCT m, edge""",
                params,
                timeout=10000,
            )

            next_frontier: set[str] = set()
            for row in result.result_set:
                entity = self._node_to_entity(row[0], group_id)
                rel = self._edge_to_relationship(row[1])

                if rel.id not in seen_edges:
                    seen_edges.add(rel.id)
                    results.append((entity, rel))

                if entity.id not in seen_entities:
                    seen_entities.add(entity.id)
                    next_frontier.add(entity.id)

            frontier = next_frontier

        return results

    async def get_all_edges(
        self,
        group_id: str,
        entity_ids: set[str] | None = None,
        limit: int = 10000,
    ) -> list[Relationship]:
        """Return all active edges for a group, optionally filtered to a set of entity IDs."""
        now_str = datetime.utcnow().isoformat()
        conditions = [
            "(edge.valid_to IS NULL OR edge.valid_to > $now)",
            "edge.group_id = $group_id",
        ]
        params: dict = {"now": now_str, "group_id": group_id, "limit": limit}

        if entity_ids is not None:
            ids_list = list(entity_ids)
            conditions.append("a.id IN $ids AND b.id IN $ids")
            params["ids"] = ids_list

        where = " AND ".join(conditions)
        result = await self._query(
            f"""MATCH (a:Entity)-[edge:RELATES_TO]->(b:Entity)
                WHERE {where}
                RETURN DISTINCT edge
                LIMIT $limit""",
            params,
            timeout=15000,
        )

        results = []
        for row in result.result_set:
            rel = self._edge_to_relationship(row[0])
            results.append(rel)
        return results

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str, str]]:
        group_filter = " AND r.group_id = $group_id" if group_id else ""
        params: dict = {"id": entity_id, "now": datetime.utcnow().isoformat()}
        if group_id:
            params["group_id"] = group_id

        result = await self._query(
            f"""MATCH (n:Entity {{id: $id}})-[r:RELATES_TO]-(m:Entity)
                WHERE (r.valid_to IS NULL OR r.valid_to > $now)
                  AND n.id <> m.id
                  {group_filter}
                RETURN m.id AS neighbor_id, r.weight AS weight,
                       r.predicate AS predicate, m.entity_type AS entity_type""",
            params,
        )
        return [(row[0], row[1], row[2], row[3]) for row in result.result_set]

    # --- Episodes ---

    async def create_episode(self, episode: Episode) -> str:
        content = self._encrypt(episode.group_id, episode.content)
        now_iso = datetime.utcnow().isoformat()
        status_val = episode.status.value if hasattr(episode.status, "value") else episode.status
        await self._query(
            """CREATE (n:Episode {
                id: $id, content: $content, source: $source,
                status: $status, group_id: $group_id,
                session_id: $session_id, created_at: $created_at,
                updated_at: $updated_at, error: $error,
                retry_count: $retry_count,
                processing_duration_ms: $processing_duration_ms,
                encoding_context: $encoding_context
            })""",
            {
                "id": episode.id,
                "content": content,
                "source": episode.source,
                "status": status_val,
                "group_id": episode.group_id,
                "session_id": episode.session_id,
                "created_at": (episode.created_at.isoformat() if episode.created_at else now_iso),
                "updated_at": (episode.updated_at.isoformat() if episode.updated_at else now_iso),
                "error": episode.error,
                "retry_count": episode.retry_count,
                "processing_duration_ms": episode.processing_duration_ms,
                "encoding_context": episode.encoding_context,
            },
        )
        return episode.id

    async def update_episode(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        updates["updated_at"] = datetime.utcnow().isoformat()
        invalid = set(updates.keys()) - EPISODE_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed episode update fields: {invalid}")
        set_parts = []
        for key in updates:
            set_parts.append(f"n.{key} = ${key}")
        set_clause = ", ".join(set_parts)
        params = dict(updates)
        params["id"] = episode_id
        params["gid"] = group_id
        await self._query(
            f"MATCH (n:Episode {{id: $id, group_id: $gid}}) SET {set_clause}",
            params,
        )

    async def get_episodes(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]:
        if group_id:
            result = await self._query(
                """MATCH (n:Episode {group_id: $gid})
                   RETURN n ORDER BY n.created_at DESC SKIP $offset LIMIT $limit""",
                {"gid": group_id, "offset": offset, "limit": limit},
            )
        else:
            result = await self._query(
                "MATCH (n:Episode) RETURN n ORDER BY n.created_at DESC SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": limit},
            )
        return [self._node_to_episode(row[0], group_id) for row in result.result_set]

    async def get_episode_by_id(self, episode_id: str, group_id: str) -> Episode | None:
        result = await self._query(
            "MATCH (n:Episode {id: $id, group_id: $gid}) RETURN n",
            {"id": episode_id, "gid": group_id},
        )
        if not result.result_set:
            return None
        return self._node_to_episode(result.result_set[0][0], group_id)

    async def get_episode_entities(self, episode_id: str) -> list[str]:
        result = await self._query(
            "MATCH (ep:Episode {id: $eid})-[:HAS_ENTITY]->(ent:Entity) RETURN ent.id",
            {"eid": episode_id},
        )
        return [row[0] for row in result.result_set]

    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        await self._query(
            """MATCH (ep:Episode {id: $eid}), (ent:Entity {id: $entid})
               MERGE (ep)-[:HAS_ENTITY]->(ent)""",
            {"eid": episode_id, "entid": entity_id},
        )

    async def get_stats(self, group_id: str | None = None) -> dict:
        if group_id:
            ent_result = await self._query(
                "MATCH (n:Entity {group_id: $gid}) WHERE n.deleted_at IS NULL RETURN COUNT(n)",
                {"gid": group_id},
            )
            rel_result = await self._query(
                "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() RETURN COUNT(r)",
                {"gid": group_id},
            )
            ep_result = await self._query(
                "MATCH (n:Episode {group_id: $gid}) RETURN COUNT(n)",
                {"gid": group_id},
            )
        else:
            ent_result = await self._query(
                "MATCH (n:Entity) WHERE n.deleted_at IS NULL RETURN COUNT(n)"
            )
            rel_result = await self._query("MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r)")
            ep_result = await self._query("MATCH (n:Episode) RETURN COUNT(n)")

        return {
            "entities": ent_result.result_set[0][0] if ent_result.result_set else 0,
            "relationships": rel_result.result_set[0][0] if rel_result.result_set else 0,
            "episodes": ep_result.result_set[0][0] if ep_result.result_set else 0,
        }

    # --- Extra API methods (used by REST endpoints) ---

    async def get_episodes_paginated(
        self,
        group_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        source: str | None = None,
        status: str | None = None,
    ) -> tuple[list[Episode], str | None]:
        conditions: list[str] = []
        params: dict = {"limit": limit + 1}

        if group_id:
            conditions.append("n.group_id = $group_id")
            params["group_id"] = group_id
        if source:
            conditions.append("n.source = $source")
            params["source"] = source
        if status:
            conditions.append("n.status = $status")
            params["status"] = status
        if cursor:
            conditions.append("n.created_at < $cursor")
            params["cursor"] = cursor

        where = " AND ".join(conditions) if conditions else "TRUE"
        result = await self._query(
            f"""MATCH (n:Episode)
                WHERE {where}
                RETURN n ORDER BY n.created_at DESC LIMIT $limit""",
            params,
        )

        episodes = [self._node_to_episode(row[0], group_id) for row in result.result_set[:limit]]
        next_cursor = None
        if len(result.result_set) > limit:
            next_cursor = episodes[-1].created_at.isoformat()

        return episodes, next_cursor

    async def get_top_connected(self, group_id: str | None = None, limit: int = 10) -> list[dict]:
        group_filter = " AND e.group_id = $group_id" if group_id else ""
        params: dict = {"limit": limit, "now": datetime.utcnow().isoformat()}
        if group_id:
            params["group_id"] = group_id

        result = await self._query(
            f"""MATCH (e:Entity)
                WHERE e.deleted_at IS NULL{group_filter}
                OPTIONAL MATCH (e)-[r:RELATES_TO]-()
                WHERE r.valid_to IS NULL OR r.valid_to > $now
                RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                       COUNT(r) AS edge_count
                ORDER BY edge_count DESC
                LIMIT $limit""",
            params,
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "entityType": row[2],
                "edgeCount": row[3],
            }
            for row in result.result_set
        ]

    async def get_growth_timeline(self, group_id: str | None = None, days: int = 30) -> list[dict]:
        """Return daily entity and episode counts. Aggregated in Python."""
        from datetime import timedelta

        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        params: dict = {"since": since}

        ep_filter = "n.created_at >= $since"
        ent_filter = "n.created_at >= $since AND n.deleted_at IS NULL"
        if group_id:
            ep_filter += " AND n.group_id = $group_id"
            ent_filter += " AND n.group_id = $group_id"
            params["group_id"] = group_id

        ep_result = await self._query(
            f"MATCH (n:Episode) WHERE {ep_filter} RETURN n.created_at",
            params,
        )
        ent_result = await self._query(
            f"MATCH (n:Entity) WHERE {ent_filter} RETURN n.created_at",
            params,
        )

        # Aggregate by date in Python
        ep_map: dict[str, int] = {}
        for row in ep_result.result_set:
            day = row[0][:10] if row[0] else None
            if day:
                ep_map[day] = ep_map.get(day, 0) + 1

        ent_map: dict[str, int] = {}
        for row in ent_result.result_set:
            day = row[0][:10] if row[0] else None
            if day:
                ent_map[day] = ent_map.get(day, 0) + 1

        all_days = sorted(set(list(ep_map.keys()) + list(ent_map.keys())), reverse=True)
        return [
            {
                "date": day,
                "episodes": ep_map.get(day, 0),
                "entities": ent_map.get(day, 0),
            }
            for day in all_days[:days]
        ]

    async def get_entity_type_counts(self, group_id: str | None = None) -> dict[str, int]:
        if group_id:
            result = await self._query(
                """MATCH (n:Entity {group_id: $gid})
                   WHERE n.deleted_at IS NULL
                   RETURN n.entity_type, COUNT(n)""",
                {"gid": group_id},
            )
        else:
            result = await self._query(
                """MATCH (n:Entity)
                   WHERE n.deleted_at IS NULL
                   RETURN n.entity_type, COUNT(n)""",
            )
        return {row[0]: row[1] for row in result.result_set}

    # --- Consolidation methods ---

    async def get_co_occurring_entity_pairs(
        self,
        group_id: str,
        since: datetime | None = None,
        min_co_occurrence: int = 3,
        limit: int = 100,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs that co-occur in episodes but lack a relationship."""
        since_clause = ""
        params: dict = {"gid": group_id, "min_co": min_co_occurrence, "limit": limit,
                       "now": datetime.utcnow().isoformat()}
        if since:
            since_clause = "AND ep.created_at >= $since"
            params["since"] = since.isoformat()

        cypher = f"""
            MATCH (ep:Episode)-[:HAS_ENTITY]->(e1:Entity),
                  (ep)-[:HAS_ENTITY]->(e2:Entity)
            WHERE e1.group_id = $gid AND e2.group_id = $gid
              AND e1.deleted_at IS NULL AND e2.deleted_at IS NULL
              AND e1.id < e2.id
              {since_clause}
            WITH e1, e2, COUNT(DISTINCT ep) AS cnt
            WHERE cnt >= $min_co
            OPTIONAL MATCH (e1)-[r:RELATES_TO]-(e2)
            WHERE r.valid_to IS NULL OR r.valid_to > $now
            WITH e1, e2, cnt, r
            WHERE r IS NULL
            RETURN e1.id, e2.id, cnt
            ORDER BY cnt DESC
            LIMIT $limit
        """
        result = await self._query(cypher, params)
        return [(row[0], row[1], row[2]) for row in result.result_set]

    async def get_entity_episode_counts(
        self,
        group_id: str,
        entity_ids: list[str],
    ) -> dict[str, int]:
        """Return how many episodes each entity appears in."""
        if not entity_ids:
            return {}
        result = await self._query(
            """MATCH (ep:Episode)-[:HAS_ENTITY]->(e:Entity)
               WHERE e.group_id = $gid AND e.id IN $ids
               RETURN e.id, COUNT(DISTINCT ep) AS cnt""",
            {"gid": group_id, "ids": entity_ids},
        )
        return {row[0]: row[1] for row in result.result_set}

    async def get_dead_entities(
        self,
        group_id: str,
        min_age_days: int = 30,
        limit: int = 100,
        max_access_count: int = 0,
    ) -> list[Entity]:
        """Find entities with no relationships and low access."""
        cutoff = (datetime.utcnow() - timedelta(days=min_age_days)).isoformat()
        result = await self._query(
            """MATCH (n:Entity {group_id: $gid})
               WHERE n.deleted_at IS NULL
                 AND n.access_count <= $max_access
                 AND n.created_at < $cutoff
                 AND (n.identity_core IS NULL OR n.identity_core = false)
               OPTIONAL MATCH (n)-[r:RELATES_TO]-()
               WHERE r.valid_to IS NULL OR r.valid_to > $now
               WITH n, r
               WHERE r IS NULL
               RETURN n
               ORDER BY n.access_count ASC, n.created_at ASC
               LIMIT $limit""",
            {
                "gid": group_id, "cutoff": cutoff, "max_access": max_access_count,
                "limit": limit, "now": datetime.utcnow().isoformat(),
            },
        )
        return [self._node_to_entity(row[0], group_id) for row in result.result_set]

    async def get_identity_core_entities(self, group_id: str) -> list[Entity]:
        """Return all entities marked as identity_core for a group."""
        result = await self._query(
            """MATCH (n:Entity {group_id: $gid})
               WHERE n.deleted_at IS NULL AND n.identity_core = true
               RETURN n""",
            {"gid": group_id},
        )
        return [self._node_to_entity(row[0], group_id) for row in result.result_set]

    async def merge_entities(
        self,
        keep_id: str,
        remove_id: str,
        group_id: str,
    ) -> int:
        """Merge remove_id into keep_id: re-point edges, merge summaries, soft-delete loser.

        FalkorDB edges are structural — can't UPDATE endpoints. Strategy:
        fetch old edges, CREATE new edges from/to keep_id, DELETE old edges.
        """
        transferred = 0

        # 1. Fetch outgoing RELATES_TO from remove_id (excluding keep_id as target)
        out_result = await self._query(
            """MATCH (old:Entity {id: $remove_id})-[r:RELATES_TO]->(t:Entity)
               WHERE r.group_id = $gid AND t.id <> $keep_id
               RETURN r, t.id""",
            {"remove_id": remove_id, "gid": group_id, "keep_id": keep_id},
        )
        for row in out_result.result_set:
            old_rel = self._edge_to_relationship(row[0])
            target_id = row[1]
            # Check if keeper already has the same edge — skip if duplicate
            existing = await self.find_existing_relationship(
                keep_id, target_id, old_rel.predicate, group_id,
            )
            if existing:
                # Update weight if new is higher
                if old_rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        keep_id, target_id,
                        old_rel.weight - existing.weight,
                        group_id=group_id,
                    )
            else:
                new_id = f"rel_{uuid.uuid4().hex[:12]}"
                await self._query(
                    """MATCH (k:Entity {id: $keep_id}), (t:Entity {id: $tgt_id})
                       CREATE (k)-[:RELATES_TO {
                           id: $new_id, source_id: $keep_id, target_id: $tgt_id,
                           predicate: $predicate, weight: $weight,
                           valid_from: $valid_from, valid_to: $valid_to,
                           created_at: $created_at, confidence: $confidence,
                           source_episode: $source_episode, group_id: $gid
                       }]->(t)""",
                    {
                        "keep_id": keep_id,
                        "tgt_id": target_id,
                        "new_id": new_id,
                        "predicate": old_rel.predicate,
                        "weight": old_rel.weight,
                        "valid_from": (
                            old_rel.valid_from.isoformat()
                            if old_rel.valid_from else None
                        ),
                        "valid_to": (
                            old_rel.valid_to.isoformat()
                            if old_rel.valid_to else None
                        ),
                        "created_at": (
                            old_rel.created_at.isoformat()
                            if old_rel.created_at else None
                        ),
                        "confidence": old_rel.confidence,
                        "source_episode": old_rel.source_episode,
                        "gid": group_id,
                    },
                )
            # Delete the old edge
            await self._query(
                """MATCH ()-[r:RELATES_TO {id: $rid}]->()
                   WHERE r.group_id = $gid DELETE r""",
                {"rid": old_rel.id, "gid": group_id},
            )
            transferred += 1

        # 2. Fetch incoming RELATES_TO to remove_id (excluding keep_id as source)
        in_result = await self._query(
            """MATCH (s:Entity)-[r:RELATES_TO]->(old:Entity {id: $remove_id})
               WHERE r.group_id = $gid AND s.id <> $keep_id
               RETURN r, s.id""",
            {"remove_id": remove_id, "gid": group_id, "keep_id": keep_id},
        )
        for row in in_result.result_set:
            old_rel = self._edge_to_relationship(row[0])
            source_id = row[1]
            # Check if keeper already has the same incoming edge
            existing = await self.find_existing_relationship(
                source_id, keep_id, old_rel.predicate, group_id,
            )
            if existing:
                if old_rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        source_id, keep_id,
                        old_rel.weight - existing.weight,
                        group_id=group_id,
                    )
            else:
                new_id = f"rel_{uuid.uuid4().hex[:12]}"
                await self._query(
                    """MATCH (s:Entity {id: $src_id}), (k:Entity {id: $keep_id})
                       CREATE (s)-[:RELATES_TO {
                           id: $new_id, source_id: $src_id, target_id: $keep_id,
                           predicate: $predicate, weight: $weight,
                           valid_from: $valid_from, valid_to: $valid_to,
                           created_at: $created_at, confidence: $confidence,
                           source_episode: $source_episode, group_id: $gid
                       }]->(k)""",
                    {
                        "src_id": source_id,
                        "keep_id": keep_id,
                        "new_id": new_id,
                        "predicate": old_rel.predicate,
                        "weight": old_rel.weight,
                        "valid_from": (
                            old_rel.valid_from.isoformat()
                            if old_rel.valid_from else None
                        ),
                        "valid_to": (
                            old_rel.valid_to.isoformat()
                            if old_rel.valid_to else None
                        ),
                        "created_at": (
                            old_rel.created_at.isoformat()
                            if old_rel.created_at else None
                        ),
                        "confidence": old_rel.confidence,
                        "source_episode": old_rel.source_episode,
                        "gid": group_id,
                    },
                )
            await self._query(
                """MATCH ()-[r:RELATES_TO {id: $rid}]->()
                   WHERE r.group_id = $gid DELETE r""",
                {"rid": old_rel.id, "gid": group_id},
            )
            transferred += 1

        # 3. Re-point HAS_ENTITY links
        await self._query(
            """MATCH (ep:Episode)-[he:HAS_ENTITY]->(old:Entity {id: $remove_id})
               MATCH (keep:Entity {id: $keep_id})
               MERGE (ep)-[:HAS_ENTITY]->(keep)
               DELETE he""",
            {"remove_id": remove_id, "keep_id": keep_id},
        )

        # 4. Merge summary + access_count on keeper
        keep_result = await self._query(
            "MATCH (n:Entity {id: $id, group_id: $gid}) RETURN n.summary, n.access_count",
            {"id": keep_id, "gid": group_id},
        )
        remove_result = await self._query(
            "MATCH (n:Entity {id: $id, group_id: $gid}) RETURN n.summary, n.access_count",
            {"id": remove_id, "gid": group_id},
        )
        if keep_result.result_set and remove_result.result_set:
            keep_summary = keep_result.result_set[0][0] or ""
            remove_summary = remove_result.result_set[0][0] or ""
            keep_count = keep_result.result_set[0][1] or 0
            remove_count = remove_result.result_set[0][1] or 0
            if remove_summary and remove_summary not in keep_summary:
                if is_meta_summary(remove_summary):
                    logger.warning(
                        "Rejected meta-contaminated summary during merge into %s: %s",
                        keep_id, remove_summary[:80],
                    )
                else:
                    merged_summary = f"{keep_summary} {remove_summary}".strip()
                    if len(merged_summary) > 500:
                        merged_summary = merged_summary[:497] + "..."
                    keep_summary = merged_summary
            await self._query(
                """MATCH (n:Entity {id: $id, group_id: $gid})
                   SET n.summary = $summary, n.access_count = $count,
                       n.updated_at = $now""",
                {
                    "id": keep_id,
                    "gid": group_id,
                    "summary": keep_summary,
                    "count": keep_count + remove_count,
                    "now": datetime.utcnow().isoformat(),
                },
            )

        # 5. Soft-delete loser
        await self._query(
            """MATCH (n:Entity {id: $id, group_id: $gid})
               SET n.deleted_at = $now""",
            {"id": remove_id, "gid": group_id, "now": datetime.utcnow().isoformat()},
        )

        return transferred

    async def path_exists_within_hops(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
        group_id: str,
    ) -> bool:
        """Check if a path exists between two entities within N hops."""
        now = datetime.utcnow().isoformat()
        cypher = (
            f"MATCH p = (s:Entity {{id: $src}})"
            f"-[:RELATES_TO*1..{max_hops}]-(t:Entity {{id: $tgt}}) "
            "WHERE ALL(r IN relationships(p) WHERE "
            "r.group_id = $gid AND "
            "(r.valid_to IS NULL OR r.valid_to > $now)) "
            "RETURN 1 LIMIT 1"
        )
        result = await self._query(
            cypher,
            {"src": source_id, "tgt": target_id, "gid": group_id, "now": now},
        )
        return bool(result.result_set)

    async def get_expired_relationships(
        self,
        group_id: str,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Return relationships whose valid_to has passed."""
        now = datetime.utcnow().isoformat()
        pred_clause = "AND r.predicate = $pred" if predicate else ""
        params: dict = {"gid": group_id, "now": now, "limit": limit}
        if predicate:
            params["pred"] = predicate
        result = await self._query(
            f"""MATCH ()-[r:RELATES_TO]->()
                WHERE r.group_id = $gid
                  AND r.valid_to IS NOT NULL
                  AND r.valid_to <= $now
                  {pred_clause}
                RETURN r LIMIT $limit""",
            params,
        )
        return [self._edge_to_relationship(row[0]) for row in result.result_set]

    async def get_relationships_by_predicate(
        self,
        group_id: str,
        predicate: str,
        active_only: bool = True,
        limit: int = 10000,
    ) -> list[Relationship]:
        """Fetch relationships matching a specific predicate."""
        active_clause = "AND (r.valid_to IS NULL OR r.valid_to > $now)" if active_only else ""
        result = await self._query(
            f"""MATCH ()-[r:RELATES_TO]->()
                WHERE r.group_id = $gid AND r.predicate = $pred
                  {active_clause}
                RETURN r LIMIT $limit""",
            {"gid": group_id, "pred": predicate, "limit": limit,
             "now": datetime.utcnow().isoformat()},
        )
        seen_ids: set[str] = set()
        rels = []
        for row in result.result_set:
            rel = self._edge_to_relationship(row[0])
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                rels.append(rel)
        return rels

    # --- Helpers ---

    def _node_to_entity(self, node, group_id: str | None = None) -> Entity:
        props = node.properties
        node_group = props.get("group_id", "default")
        decrypt_group = group_id or node_group
        summary = self._decrypt(decrypt_group, props.get("summary"))

        pii_categories = None
        if props.get("pii_categories"):
            try:
                pii_categories = json.loads(props["pii_categories"])
            except (json.JSONDecodeError, TypeError):
                pass

        attributes = None
        if props.get("attributes"):
            try:
                attributes = json.loads(props["attributes"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Entity(
            id=props["id"],
            name=props["name"],
            entity_type=props["entity_type"],
            summary=summary,
            attributes=attributes,
            group_id=node_group,
            created_at=_parse_dt(props.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(props.get("updated_at")) or datetime.utcnow(),
            deleted_at=_parse_dt(props.get("deleted_at")),
            activation_current=props.get("activation_current", 0.0),
            access_count=props.get("access_count", 0),
            last_accessed=_parse_dt(props.get("last_accessed")),
            pii_detected=bool(props.get("pii_detected", False)),
            pii_categories=pii_categories,
            identity_core=bool(props.get("identity_core", False)),
        )

    @staticmethod
    def _edge_to_relationship(edge) -> Relationship:
        props = edge.properties
        return Relationship(
            id=props["id"],
            source_id=props["source_id"],
            target_id=props["target_id"],
            predicate=props["predicate"],
            weight=props.get("weight", 1.0),
            valid_from=_parse_dt(props.get("valid_from")),
            valid_to=_parse_dt(props.get("valid_to")),
            created_at=_parse_dt(props.get("created_at")) or datetime.utcnow(),
            confidence=props.get("confidence", 1.0),
            source_episode=props.get("source_episode"),
            group_id=props.get("group_id", "default"),
        )

    def _node_to_episode(self, node, group_id: str | None = None) -> Episode:
        props = node.properties
        node_group = props.get("group_id", "default")
        decrypt_group = group_id or node_group
        content = self._decrypt(decrypt_group, props.get("content", ""))

        return Episode(
            id=props["id"],
            content=content or "",
            source=props.get("source"),
            status=props.get("status", "pending"),
            group_id=node_group,
            session_id=props.get("session_id"),
            created_at=_parse_dt(props.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(props.get("updated_at")),
            error=props.get("error"),
            retry_count=props.get("retry_count", 0) or 0,
            processing_duration_ms=props.get("processing_duration_ms"),
        )

    @staticmethod
    def _node_to_intention(node):
        """Convert a FalkorDB :Intention node to an Intention model."""
        from engram.models.prospective import Intention

        props = node.properties
        return Intention(
            id=props["id"],
            trigger_text=props["trigger_text"],
            action_text=props["action_text"],
            trigger_type=props.get("trigger_type", "semantic"),
            entity_name=props.get("entity_name"),
            threshold=props.get("threshold", 0.7),
            max_fires=props.get("max_fires", 5),
            fire_count=props.get("fire_count", 0),
            enabled=bool(props.get("enabled", True)),
            group_id=props.get("group_id", "default"),
            created_at=_parse_dt(props.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(props.get("updated_at")) or datetime.utcnow(),
            expires_at=_parse_dt(props.get("expires_at")),
        )

    # --- Schema Formation (Brain Architecture Phase 3) stubs ---

    async def get_schema_members(
        self, schema_entity_id: str, group_id: str,
    ) -> list[dict]:
        """Stub — FalkorDB schema members not yet implemented."""
        return []

    async def save_schema_members(
        self, schema_entity_id: str, members: list[dict], group_id: str,
    ) -> None:
        """Stub — FalkorDB schema members not yet implemented."""

    async def find_entities_by_type(
        self, entity_type: str, group_id: str, limit: int = 100,
    ) -> list[Entity]:
        """Return non-deleted entities of a specific type."""
        result = await self._query(
            """MATCH (n:Entity {entity_type: $etype, group_id: $gid})
               WHERE n.deleted_at IS NULL
               RETURN n LIMIT $limit""",
            {"etype": entity_type, "gid": group_id, "limit": limit},
        )
        return [self._node_to_entity(row[0], group_id) for row in result.result_set]

    # --- Intention methods (Wave 4 prospective memory) ---

    async def create_intention(self, intention: object) -> str:
        """Store a new intention as a :Intention node."""
        from engram.models.prospective import Intention

        i: Intention = intention  # type: ignore[assignment]
        await self._query(
            """CREATE (n:Intention {
                id: $id, trigger_text: $trigger_text, action_text: $action_text,
                trigger_type: $trigger_type, entity_name: $entity_name,
                threshold: $threshold, max_fires: $max_fires, fire_count: $fire_count,
                enabled: $enabled, group_id: $group_id,
                created_at: $created_at, updated_at: $updated_at, expires_at: $expires_at
            })""",
            {
                "id": i.id,
                "trigger_text": i.trigger_text,
                "action_text": i.action_text,
                "trigger_type": i.trigger_type,
                "entity_name": i.entity_name,
                "threshold": i.threshold,
                "max_fires": i.max_fires,
                "fire_count": i.fire_count,
                "enabled": i.enabled,
                "group_id": i.group_id,
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
                "expires_at": i.expires_at.isoformat() if i.expires_at else None,
            },
        )
        return i.id

    async def get_intention(self, id: str, group_id: str) -> object | None:
        """Get a single intention by ID and group."""
        result = await self._query(
            "MATCH (n:Intention {id: $id, group_id: $gid}) RETURN n",
            {"id": id, "gid": group_id},
        )
        if not result.result_set:
            return None
        return self._node_to_intention(result.result_set[0][0])

    async def list_intentions(
        self, group_id: str, enabled_only: bool = True,
    ) -> list:
        """List intentions for a group, filtering expired and optionally disabled."""
        if enabled_only:
            now = datetime.utcnow().isoformat()
            result = await self._query(
                """MATCH (n:Intention {group_id: $gid})
                   WHERE n.enabled = true
                     AND (n.expires_at IS NULL OR n.expires_at > $now)
                   RETURN n ORDER BY n.created_at DESC""",
                {"gid": group_id, "now": now},
            )
        else:
            result = await self._query(
                """MATCH (n:Intention {group_id: $gid})
                   RETURN n ORDER BY n.created_at DESC""",
                {"gid": group_id},
            )
        return [self._node_to_intention(row[0]) for row in result.result_set]

    async def update_intention(
        self, id: str, updates: dict, group_id: str,
    ) -> None:
        """Update intention fields."""
        allowed = {"trigger_text", "action_text", "threshold", "max_fires", "enabled", "expires_at"}
        set_parts = []
        params: dict = {"id": id, "gid": group_id}
        for key, val in updates.items():
            if key not in allowed:
                continue
            param_name = f"u_{key}"
            set_parts.append(f"n.{key} = ${param_name}")
            params[param_name] = val
        if not set_parts:
            return
        set_parts.append("n.updated_at = $updated_at")
        params["updated_at"] = datetime.utcnow().isoformat()
        cypher = (
            f"MATCH (n:Intention {{id: $id, group_id: $gid}}) "
            f"SET {', '.join(set_parts)}"
        )
        await self._query(cypher, params)

    async def delete_intention(
        self, id: str, group_id: str, soft: bool = True,
    ) -> None:
        """Delete an intention (soft = disable, hard = remove)."""
        if soft:
            await self._query(
                """MATCH (n:Intention {id: $id, group_id: $gid})
                   SET n.enabled = false, n.updated_at = $now""",
                {"id": id, "gid": group_id, "now": datetime.utcnow().isoformat()},
            )
        else:
            await self._query(
                "MATCH (n:Intention {id: $id, group_id: $gid}) DELETE n",
                {"id": id, "gid": group_id},
            )

    async def increment_intention_fire_count(
        self, id: str, group_id: str,
    ) -> None:
        """Increment fire_count by 1."""
        await self._query(
            """MATCH (n:Intention {id: $id, group_id: $gid})
               SET n.fire_count = n.fire_count + 1, n.updated_at = $now""",
            {"id": id, "gid": group_id, "now": datetime.utcnow().isoformat()},
        )
