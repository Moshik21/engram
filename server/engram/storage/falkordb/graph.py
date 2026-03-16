"""FalkorDB implementation of GraphStore protocol (Full mode)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from engram.config import FalkorDBConfig
from engram.entity_dedup_policy import NameRegime, analyze_name, entity_identifier_facets
from engram.models.entity import Entity
from engram.models.episode import Attachment, Episode
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.storage.protocols import ENTITY_UPDATABLE_FIELDS, EPISODE_UPDATABLE_FIELDS
from engram.utils.dates import utc_now, utc_now_iso
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from engram.models.prospective import Intention


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _falkordb_password_candidates(config: FalkorDBConfig) -> list[str | None]:
    """Return passwords to try, preferring explicit configuration first."""
    configured = config.password or None
    candidates: list[str | None] = [configured]
    if configured is None:
        fallback = os.environ.get("ENGRAM_FALKORDB__PASSWORD", "engram_dev")
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _evidence_node_to_dict(node) -> dict:
    """Convert a FalkorDB :Evidence node to the GraphStore evidence shape."""
    props = node.properties
    return {
        "evidence_id": props["evidence_id"],
        "episode_id": props["episode_id"],
        "group_id": props.get("group_id", "default"),
        "fact_class": props["fact_class"],
        "confidence": props.get("confidence", 0.0),
        "source_type": props.get("source_type", ""),
        "extractor_name": props.get("extractor_name", ""),
        "payload": json.loads(props.get("payload_json", "{}") or "{}"),
        "source_span": props.get("source_span"),
        "corroborating_signals": json.loads(props.get("signals_json", "[]") or "[]"),
        "ambiguity_tags": json.loads(
            props.get("ambiguity_tags_json", "[]") or "[]",
        ),
        "ambiguity_score": props.get("ambiguity_score", 0.0) or 0.0,
        "adjudication_request_id": props.get("adjudication_request_id"),
        "status": props.get("status", "pending"),
        "commit_reason": props.get("commit_reason"),
        "committed_id": props.get("committed_id"),
        "deferred_cycles": props.get("deferred_cycles", 0) or 0,
        "created_at": props.get("created_at"),
        "resolved_at": props.get("resolved_at"),
    }


def _adjudication_node_to_dict(node) -> dict:
    """Convert a FalkorDB :AdjudicationRequest node to storage shape."""
    props = node.properties
    return {
        "request_id": props["request_id"],
        "episode_id": props["episode_id"],
        "group_id": props.get("group_id", "default"),
        "status": props.get("status", "pending"),
        "ambiguity_tags": json.loads(
            props.get("ambiguity_tags_json", "[]") or "[]",
        ),
        "evidence_ids": json.loads(props.get("evidence_ids_json", "[]") or "[]"),
        "selected_text": props.get("selected_text", "") or "",
        "request_reason": props.get("request_reason", "") or "",
        "resolution_source": props.get("resolution_source"),
        "resolution_payload": json.loads(
            props.get("resolution_payload_json", "null") or "null",
        ),
        "attempt_count": props.get("attempt_count", 0) or 0,
        "created_at": props.get("created_at"),
        "resolved_at": props.get("resolved_at"),
    }


class FalkorDBGraphStore:
    """Graph store backed by FalkorDB (Redis-based graph database)."""

    def __init__(self, config: FalkorDBConfig, encryptor=None) -> None:
        self._config = config
        self._encryptor = encryptor
        self._db: Any | None = None
        self._graph: Any | None = None

    async def _query(self, cypher: str, params: dict | None = None, timeout: int | None = None):
        """Execute a Cypher query via thread pool (falkordb is synchronous)."""
        graph = self._graph
        if graph is None:
            raise RuntimeError("FalkorDB graph store not initialized")
        kwargs: dict = {"params": params or {}}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await asyncio.to_thread(cast(Any, graph.query), cypher, **kwargs)

    async def initialize(self) -> None:
        """Connect to FalkorDB and create indexes."""
        from falkordb import FalkorDB  # type: ignore[import-untyped]

        last_error: Exception | None = None
        for password in _falkordb_password_candidates(self._config):
            kwargs: dict = {
                "host": self._config.host,
                "port": self._config.port,
                "password": password,
            }
            if self._config.ssl:
                kwargs["ssl"] = True
                if self._config.ssl_ca_cert:
                    kwargs["ssl_ca_certs"] = self._config.ssl_ca_cert
            try:
                self._db = cast(Any, await asyncio.to_thread(FalkorDB, **kwargs))
                self._graph = cast(
                    Any,
                    await asyncio.to_thread(
                        cast(Any, self._db.select_graph),
                        self._config.graph_name,
                    ),
                )
                break
            except Exception as exc:
                last_error = exc
                self._db = None
                self._graph = None
        else:
            assert last_error is not None
            raise last_error

        # Create indexes — FalkorDB errors on duplicate, so wrap each in try/except
        indexes = [
            "CREATE INDEX FOR (n:Entity) ON (n.id)",
            "CREATE INDEX FOR (n:Entity) ON (n.group_id)",
            "CREATE INDEX FOR (n:Entity) ON (n.name)",
            "CREATE INDEX FOR (n:Entity) ON (n.entity_type)",
            "CREATE INDEX FOR (n:Entity) ON (n.lexical_regime)",
            "CREATE INDEX FOR (n:Entity) ON (n.canonical_identifier)",
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
            "CREATE INDEX FOR (n:Evidence) ON (n.evidence_id)",
            "CREATE INDEX FOR (n:Evidence) ON (n.group_id)",
            "CREATE INDEX FOR (n:Evidence) ON (n.status)",
            "CREATE INDEX FOR (n:Evidence) ON (n.episode_id)",
            "CREATE INDEX FOR (n:Evidence) ON (n.adjudication_request_id)",
            "CREATE INDEX FOR (n:AdjudicationRequest) ON (n.request_id)",
            "CREATE INDEX FOR (n:AdjudicationRequest) ON (n.group_id)",
            "CREATE INDEX FOR (n:AdjudicationRequest) ON (n.status)",
            "CREATE INDEX FOR (n:AdjudicationRequest) ON (n.episode_id)",
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
        return cast(str | None, self._encryptor.encrypt(group_id, plaintext))

    def _decrypt(self, group_id: str, data: str | None) -> str | None:
        if not data or not self._encryptor:
            return data
        return cast(str | None, self._encryptor.decrypt(group_id, data))

    # --- Entities ---

    async def create_entity(self, entity: Entity) -> str:
        now = utc_now_iso()
        summary = self._encrypt(entity.group_id, entity.summary)
        await self._query(
            """CREATE (n:Entity {
                id: $id, name: $name, entity_type: $entity_type,
                summary: $summary, attributes: $attributes,
                group_id: $group_id,
                created_at: $created_at, updated_at: $updated_at,
                activation_current: $activation_current,
                access_count: $access_count, last_accessed: $last_accessed,
                pii_detected: $pii_detected, pii_categories: $pii_categories,
                lexical_regime: $lexical_regime,
                canonical_identifier: $canonical_identifier,
                identifier_label: $identifier_label
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
                "lexical_regime": entity.lexical_regime,
                "canonical_identifier": entity.canonical_identifier,
                "identifier_label": entity.identifier_label,
            },
        )
        return entity.id

    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None:
        result = await self._query(
            """MATCH (n:Entity {id: $id, group_id: $gid})
               WHERE n.deleted_at IS NULL OR n.merged_into IS NOT NULL
               RETURN n""",
            {"id": entity_id, "gid": group_id},
        )
        if not result.result_set:
            return None
        node = result.result_set[0][0]
        return self._node_to_entity(node, group_id)

    async def batch_get_entities(
        self,
        entity_ids: list[str],
        group_id: str,
    ) -> dict[str, Entity]:
        if not entity_ids:
            return {}
        result = await self._query(
            """MATCH (n:Entity)
               WHERE n.id IN $ids AND n.group_id = $gid AND n.deleted_at IS NULL
               RETURN n""",
            {"ids": entity_ids, "gid": group_id},
        )
        entities: dict[str, Entity] = {}
        for row in result.result_set:
            entity = self._node_to_entity(row[0], group_id)
            entities[entity.id] = entity
        return entities

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        if not updates:
            return
        if "name" in updates or "entity_type" in updates:
            current = await self.get_entity(entity_id, group_id)
            if current is not None:
                next_name = updates.get("name", current.name)
                facets = entity_identifier_facets(next_name)
                updates["lexical_regime"] = facets["lexical_regime"]
                updates["canonical_identifier"] = facets["canonical_identifier"]
                updates["identifier_label"] = bool(facets["identifier_label"])
        updates["updated_at"] = utc_now_iso()
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
                {"id": entity_id, "gid": group_id, "deleted_at": utc_now_iso()},
            )
        else:
            await self._query(
                "MATCH (n:Entity {id: $id, group_id: $gid}) DETACH DELETE n",
                {"id": entity_id, "gid": group_id},
            )

    async def delete_group(self, group_id: str) -> None:
        """Delete all nodes and relationships belonging to *group_id*.

        Removes Episode, EpisodeCue, Entity, and Relationship nodes
        (plus all edges connected to them) in a single DETACH DELETE
        per label to avoid oversized transactions.
        """
        for label in ("EpisodeCue", "Episode", "Relationship", "Intention", "Entity"):
            try:
                await self._query(
                    f"MATCH (n:{label} {{group_id: $gid}}) DETACH DELETE n",
                    {"gid": group_id},
                )
            except Exception:
                logger.debug(
                    "delete_group: failed to delete %s nodes for %s",
                    label,
                    group_id,
                    exc_info=True,
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
        self,
        name: str,
        group_id: str,
        limit: int = 30,
    ) -> list[Entity]:
        """Retrieve candidate entities for fuzzy resolution via CONTAINS search."""
        seen_ids: set[str] = set()
        results: list[Entity] = []
        form = analyze_name(name)
        regime = form.regime

        # Phase 1: Exact canonical identifier match
        if form.canonical_code:
            canonical_result = await self._query(
                """MATCH (n:Entity)
                   WHERE n.canonical_identifier = $canonical_identifier
                     AND n.group_id = $gid AND n.deleted_at IS NULL
                   RETURN n LIMIT $limit""",
                {
                    "canonical_identifier": form.canonical_code,
                    "gid": group_id,
                    "limit": limit - len(results),
                },
            )
            for row in canonical_result.result_set:
                entity = self._node_to_entity(row[0], group_id)
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 2: Full name CONTAINS match (indexed on n.name)
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

        # Phase 3: Token fallback — search individual tokens >= 3 chars
        tokens = [t for t in name.strip().split() if len(t) >= 3]
        if regime != NameRegime.NATURAL_LANGUAGE:
            tokens = []
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
                   source_episode: $source_episode, group_id: $group_id,
                   polarity: $polarity
               }]->(t)""",
            {
                "id": rel.id,
                "src": rel.source_id,
                "tgt": rel.target_id,
                "predicate": rel.predicate,
                "weight": rel.weight,
                "valid_from": rel.valid_from.isoformat() if rel.valid_from else None,
                "valid_to": rel.valid_to.isoformat() if rel.valid_to else None,
                "created_at": (rel.created_at.isoformat() if rel.created_at else utc_now_iso()),
                "confidence": rel.confidence,
                "source_episode": rel.source_episode,
                "group_id": rel.group_id,
                "polarity": rel.polarity,
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
            match = "MATCH (s:Entity)-[r:RELATES_TO]-(t:Entity) WHERE (s.id = $eid OR t.id = $eid)"

        conditions = []
        params: dict = {"eid": entity_id, "now": utc_now_iso()}
        if predicate:
            conditions.append("r.predicate = $predicate")
            params["predicate"] = predicate
        if active_only:
            conditions.append("(r.valid_to IS NULL OR r.valid_to > $now)")
        conditions.append("r.group_id = $gid")
        params["gid"] = group_id
        # Filter out edges to soft-deleted entities
        conditions.append("s.deleted_at IS NULL")
        conditions.append("t.deleted_at IS NULL")

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
        effective_valid_to = valid_to
        now = utc_now()
        if effective_valid_to >= now:
            effective_valid_to = now - timedelta(microseconds=1)
        await self._query(
            """MATCH ()-[r:RELATES_TO {id: $id}]->()
               WHERE r.group_id = $gid
               SET r.valid_to = $valid_to""",
            {
                "id": rel_id,
                "gid": group_id,
                "valid_to": effective_valid_to.isoformat(),
            },
        )

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        weight_delta: float,
        max_weight: float = 3.0,
        group_id: str = "default",
        predicate: str | None = None,
    ) -> float | None:
        """Atomically increment edge weight in FalkorDB, capped at max_weight."""
        predicate_clause = " AND r.predicate = $predicate" if predicate else ""
        params = {
            "src": source_id,
            "tgt": target_id,
            "delta": weight_delta,
            "max_w": max_weight,
            "gid": group_id,
            "now": utc_now_iso(),
        }
        if predicate:
            params["predicate"] = predicate
        result = await self._query(
            """MATCH (s:Entity)-[r:RELATES_TO]-(t:Entity)
               WHERE s.id = $src AND t.id = $tgt
                 AND r.group_id = $gid
                 AND (r.valid_to IS NULL OR r.valid_to > $now)
            """
            + predicate_clause
            + """
               SET r.weight = CASE
                   WHEN r.weight + $delta > $max_w THEN $max_w
                   ELSE r.weight + $delta
               END
               RETURN r.weight""",
            params,
        )
        if result.result_set:
            return float(result.result_set[0][0])
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
        now = utc_now_iso()
        result = await self._query(
            """MATCH (s:Entity {id: $src})-[r:RELATES_TO]->(t:Entity {id: $tgt})
               WHERE r.predicate = $predicate
                 AND r.group_id = $gid
                 AND (r.valid_to IS NULL OR r.valid_to > $now)
               RETURN r LIMIT 1""",
            {
                "src": source_id,
                "tgt": target_id,
                "predicate": predicate,
                "gid": group_id,
                "now": now,
            },
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
        # Filter out edges to soft-deleted entities
        time_conditions.append("s.deleted_at IS NULL")
        time_conditions.append("t.deleted_at IS NULL")
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
        max_results: int = 5000,
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops via iterative BFS (avoids combinatorial explosion)."""
        now_str = utc_now_iso()
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

                if len(results) >= max_results:
                    break

            if len(results) >= max_results:
                break

            frontier = next_frontier

        return results

    async def get_all_edges(
        self,
        group_id: str,
        entity_ids: set[str] | None = None,
        limit: int = 10000,
    ) -> list[Relationship]:
        """Return all active edges for a group, optionally filtered to a set of entity IDs."""
        now_str = utc_now_iso()
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
        params: dict = {"id": entity_id, "now": utc_now_iso()}
        if group_id:
            params["group_id"] = group_id

        result = await self._query(
            f"""MATCH (n:Entity {{id: $id}})-[r:RELATES_TO]-(m:Entity)
                WHERE (r.valid_to IS NULL OR r.valid_to > $now)
                  AND coalesce(r.polarity, 'positive') <> 'negative'
                  AND n.id <> m.id
                  AND m.deleted_at IS NULL
                  {group_filter}
                RETURN m.id AS neighbor_id,
                       CASE
                           WHEN coalesce(r.polarity, 'positive') = 'uncertain'
                               THEN r.weight * 0.5
                           ELSE r.weight
                       END AS weight,
                       r.predicate AS predicate, m.entity_type AS entity_type""",
            params,
        )
        return [(row[0], row[1], row[2], row[3]) for row in result.result_set]

    # --- Episodes ---

    async def create_episode(self, episode: Episode) -> str:
        content = self._encrypt(episode.group_id, episode.content)
        now_iso = utc_now_iso()
        status_val = episode.status.value if hasattr(episode.status, "value") else episode.status
        await self._query(
            """CREATE (n:Episode {
                id: $id, content: $content, source: $source,
                status: $status, group_id: $group_id,
                session_id: $session_id, created_at: $created_at,
                updated_at: $updated_at, error: $error,
                retry_count: $retry_count,
                processing_duration_ms: $processing_duration_ms,
                encoding_context: $encoding_context,
                memory_tier: $memory_tier,
                consolidation_cycles: $consolidation_cycles,
                entity_coverage: $entity_coverage,
                projection_state: $projection_state,
                last_projection_reason: $last_projection_reason,
                last_projected_at: $last_projected_at,
                conversation_date: $conversation_date,
                attachments_json: $attachments_json
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
                "memory_tier": episode.memory_tier,
                "consolidation_cycles": episode.consolidation_cycles,
                "entity_coverage": episode.entity_coverage,
                "projection_state": (
                    episode.projection_state.value
                    if hasattr(episode.projection_state, "value")
                    else episode.projection_state
                ),
                "last_projection_reason": episode.last_projection_reason,
                "last_projected_at": (
                    episode.last_projected_at.isoformat() if episode.last_projected_at else None
                ),
                "conversation_date": (
                    episode.conversation_date.isoformat() if episode.conversation_date else None
                ),
                "attachments_json": (
                    json.dumps([a.model_dump() for a in episode.attachments])
                    if episode.attachments
                    else "[]"
                ),
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
        updates["updated_at"] = utc_now_iso()
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

    async def get_episodes_for_entity(
        self,
        entity_id: str,
        group_id: str = "default",
        limit: int = 20,
    ) -> list[str]:
        """Return episode IDs linked to an entity, newest first."""
        result = await self._query(
            """MATCH (ep:Episode {group_id: $gid})-[:HAS_ENTITY]->(ent:Entity {id: $eid})
               RETURN ep.id
               ORDER BY ep.created_at DESC
               LIMIT $lim""",
            {"eid": entity_id, "gid": group_id, "lim": limit},
        )
        return [row[0] for row in result.result_set]

    async def get_adjacent_episodes(
        self,
        episode_id: str,
        group_id: str,
        limit: int = 3,
    ) -> list[Episode]:
        """Get temporally adjacent episodes from the same session."""
        result = await self._query(
            """MATCH (ref:Episode {id: $id, group_id: $gid})
               WHERE ref.session_id IS NOT NULL
               MATCH (ep:Episode {session_id: ref.session_id, group_id: $gid})
               WHERE ep.id <> $id
               RETURN ep
               ORDER BY ep.created_at
               LIMIT $lim""",
            {"id": episode_id, "gid": group_id, "lim": limit},
        )
        episodes = []
        for record in result.result_set:
            episodes.append(self._node_to_episode(record[0], group_id))
        return episodes

    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        await self._query(
            """MATCH (ep:Episode {id: $eid}), (ent:Entity {id: $entid})
               MERGE (ep)-[:HAS_ENTITY]->(ent)""",
            {"eid": episode_id, "entid": entity_id},
        )

    async def upsert_episode_cue(self, cue: EpisodeCue) -> None:
        await self._query(
            """MATCH (ep:Episode {id: $episode_id, group_id: $group_id})
               SET ep.cue_version = $cue_version,
                   ep.cue_discourse_class = $discourse_class,
                   ep.cue_projection_state = $projection_state,
                   ep.cue_score = $cue_score,
                   ep.cue_salience_score = $salience_score,
                   ep.cue_projection_priority = $projection_priority,
                   ep.cue_route_reason = $route_reason,
                   ep.cue_text = $cue_text,
                   ep.cue_entity_mentions_json = $entity_mentions_json,
                   ep.cue_temporal_markers_json = $temporal_markers_json,
                   ep.cue_quote_spans_json = $quote_spans_json,
                   ep.cue_contradiction_keys_json = $contradiction_keys_json,
                   ep.cue_first_spans_json = $first_spans_json,
                   ep.cue_hit_count = $hit_count,
                   ep.cue_surfaced_count = $surfaced_count,
                   ep.cue_selected_count = $selected_count,
                   ep.cue_used_count = $used_count,
                   ep.cue_near_miss_count = $near_miss_count,
                   ep.cue_policy_score = $policy_score,
                   ep.cue_projection_attempts = $projection_attempts,
                   ep.cue_last_hit_at = $last_hit_at,
                   ep.cue_last_feedback_at = $last_feedback_at,
                   ep.cue_last_projected_at = $last_projected_at,
                   ep.cue_created_at = $created_at,
                   ep.cue_updated_at = $updated_at""",
            {
                "episode_id": cue.episode_id,
                "group_id": cue.group_id,
                "cue_version": cue.cue_version,
                "discourse_class": cue.discourse_class,
                "projection_state": (
                    cue.projection_state.value
                    if hasattr(cue.projection_state, "value")
                    else cue.projection_state
                ),
                "cue_score": cue.cue_score,
                "salience_score": cue.salience_score,
                "projection_priority": cue.projection_priority,
                "route_reason": cue.route_reason,
                "cue_text": cue.cue_text,
                "entity_mentions_json": json.dumps(cue.entity_mentions),
                "temporal_markers_json": json.dumps(cue.temporal_markers),
                "quote_spans_json": json.dumps(cue.quote_spans),
                "contradiction_keys_json": json.dumps(cue.contradiction_keys),
                "first_spans_json": json.dumps(cue.first_spans),
                "hit_count": cue.hit_count,
                "surfaced_count": cue.surfaced_count,
                "selected_count": cue.selected_count,
                "used_count": cue.used_count,
                "near_miss_count": cue.near_miss_count,
                "policy_score": cue.policy_score,
                "projection_attempts": cue.projection_attempts,
                "last_hit_at": cue.last_hit_at.isoformat() if cue.last_hit_at else None,
                "last_feedback_at": (
                    cue.last_feedback_at.isoformat() if cue.last_feedback_at else None
                ),
                "last_projected_at": (
                    cue.last_projected_at.isoformat() if cue.last_projected_at else None
                ),
                "created_at": cue.created_at.isoformat() if cue.created_at else None,
                "updated_at": cue.updated_at.isoformat() if cue.updated_at else None,
            },
        )

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        result = await self._query(
            "MATCH (ep:Episode {id: $episode_id, group_id: $group_id}) RETURN ep",
            {"episode_id": episode_id, "group_id": group_id},
        )
        if not result.result_set:
            return None
        props = result.result_set[0][0].properties
        if not props.get("cue_text"):
            return None
        return self._node_to_episode_cue(props, episode_id, group_id)

    async def update_episode_cue(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        remap = {
            "projection_state": "cue_projection_state",
            "cue_score": "cue_score",
            "salience_score": "cue_salience_score",
            "projection_priority": "cue_projection_priority",
            "route_reason": "cue_route_reason",
            "cue_text": "cue_text",
            "entity_mentions": "cue_entity_mentions_json",
            "temporal_markers": "cue_temporal_markers_json",
            "quote_spans": "cue_quote_spans_json",
            "contradiction_keys": "cue_contradiction_keys_json",
            "first_spans": "cue_first_spans_json",
            "hit_count": "cue_hit_count",
            "surfaced_count": "cue_surfaced_count",
            "selected_count": "cue_selected_count",
            "used_count": "cue_used_count",
            "near_miss_count": "cue_near_miss_count",
            "policy_score": "cue_policy_score",
            "projection_attempts": "cue_projection_attempts",
            "last_hit_at": "cue_last_hit_at",
            "last_feedback_at": "cue_last_feedback_at",
            "last_projected_at": "cue_last_projected_at",
        }
        params = {"episode_id": episode_id, "group_id": group_id}
        set_parts = ["ep.cue_updated_at = $cue_updated_at"]
        params["cue_updated_at"] = utc_now_iso()
        for key, value in updates.items():
            prop = remap.get(key, key)
            if key in {
                "entity_mentions",
                "temporal_markers",
                "quote_spans",
                "contradiction_keys",
                "first_spans",
            }:
                params[prop] = json.dumps(value)
            elif key == "projection_state" and hasattr(value, "value"):
                params[prop] = value.value
            elif (
                key in {"last_hit_at", "last_feedback_at", "last_projected_at"}
                and value is not None
            ):
                params[prop] = value.isoformat() if hasattr(value, "isoformat") else value
            else:
                params[prop] = value
            set_parts.append(f"ep.{prop} = ${prop}")
        await self._query(
            f"""MATCH (ep:Episode {{id: $episode_id, group_id: $group_id}})
                SET {", ".join(set_parts)}""",
            params,
        )

    async def get_stats(self, group_id: str | None = None) -> dict:
        params: dict[str, str] = {}
        entity_match = "MATCH (n:Entity)"
        entity_where = "WHERE n.deleted_at IS NULL"
        rel_match = "MATCH ()-[r:RELATES_TO]->()"
        rel_where = ""
        episode_where = ""
        if group_id:
            params["gid"] = group_id
            entity_where = "WHERE n.group_id = $gid AND n.deleted_at IS NULL"
            rel_where = "WHERE r.group_id = $gid"
            episode_where = "WHERE ep.group_id = $gid"

        ent_result = await self._query(
            f"{entity_match} {entity_where} RETURN COUNT(n)",
            params,
        )
        rel_result = await self._query(
            f"{rel_match} {rel_where} RETURN COUNT(r)",
            params,
        )
        episode_stats = await self._query(
            f"""MATCH (ep:Episode)
                {episode_where}
                RETURN COUNT(ep) AS episodes,
                       SUM(CASE WHEN ep.projection_state = 'queued' THEN 1 ELSE 0 END)
                           AS projection_queued_count,
                       SUM(CASE WHEN ep.projection_state = 'cued' THEN 1 ELSE 0 END)
                           AS projection_cued_count,
                       SUM(CASE WHEN ep.projection_state = 'cue_only' THEN 1 ELSE 0 END)
                           AS projection_cue_only_count,
                       SUM(CASE WHEN ep.projection_state = 'scheduled' THEN 1 ELSE 0 END)
                           AS projection_scheduled_count,
                       SUM(CASE WHEN ep.projection_state = 'projecting' THEN 1 ELSE 0 END)
                           AS projection_projecting_count,
                       SUM(CASE WHEN ep.projection_state = 'projected' THEN 1 ELSE 0 END)
                           AS projection_projected_count,
                       SUM(CASE WHEN ep.projection_state = 'failed' THEN 1 ELSE 0 END)
                           AS projection_failed_count,
                       SUM(CASE WHEN ep.projection_state = 'dead_letter' THEN 1 ELSE 0 END)
                           AS projection_dead_letter_count,
                       AVG(
                           CASE
                               WHEN ep.projection_state = 'projected'
                                    AND ep.processing_duration_ms IS NOT NULL
                               THEN toFloat(ep.processing_duration_ms)
                               ELSE NULL
                           END
                       ) AS avg_processing_duration_ms""",
            params,
        )
        cue_stats = await self._query(
            f"""MATCH (ep:Episode)
                {episode_where}
                RETURN SUM(
                           CASE
                               WHEN ep.cue_text IS NOT NULL AND ep.cue_text <> ''
                               THEN 1 ELSE 0
                           END
                       ) AS cue_count,
                       SUM(
                           CASE
                               WHEN coalesce(ep.cue_hit_count, 0) > 0
                               THEN 1 ELSE 0
                           END
                       ) AS cue_hit_episode_count,
                       SUM(coalesce(ep.cue_hit_count, 0)) AS cue_hit_count,
                       SUM(coalesce(ep.cue_surfaced_count, 0)) AS cue_surfaced_count,
                       SUM(coalesce(ep.cue_selected_count, 0)) AS cue_selected_count,
                       SUM(coalesce(ep.cue_used_count, 0)) AS cue_used_count,
                       SUM(coalesce(ep.cue_near_miss_count, 0)) AS cue_near_miss_count,
                       AVG(
                           CASE
                               WHEN ep.cue_text IS NOT NULL AND ep.cue_text <> ''
                               THEN toFloat(coalesce(ep.cue_policy_score, 0.0))
                               ELSE NULL
                           END
                       ) AS avg_policy_score,
                       AVG(
                           CASE
                               WHEN ep.cue_text IS NOT NULL AND ep.cue_text <> ''
                               THEN toFloat(coalesce(ep.cue_projection_attempts, 0))
                               ELSE NULL
                           END
                       ) AS avg_projection_attempts,
                       SUM(coalesce(ep.cue_projection_attempts, 0)) AS projection_attempt_total,
                       SUM(
                           CASE
                               WHEN ep.cue_projection_state = 'projected'
                               THEN 1 ELSE 0
                           END
                       ) AS projected_cue_count""",
            params,
        )
        yield_stats = await self._query(
            f"""MATCH (ep:Episode)
                {episode_where}
                WITH collect(
                    CASE
                        WHEN ep.projection_state = 'projected'
                        THEN ep.id
                        ELSE NULL
                    END
                ) AS projected_episode_ids
                OPTIONAL MATCH (pep:Episode)-[:HAS_ENTITY]->(ent:Entity)
                WHERE pep.id IN projected_episode_ids
                  AND pep.projection_state = 'projected'
                WITH projected_episode_ids, COUNT(ent) AS linked_entity_count
                OPTIONAL MATCH ()-[r:RELATES_TO]->()
                WHERE r.source_episode IN projected_episode_ids
                RETURN linked_entity_count, COUNT(r) AS relationship_count""",
            params,
        )

        entity_count = ent_result.result_set[0][0] if ent_result.result_set else 0
        relationship_count = rel_result.result_set[0][0] if rel_result.result_set else 0
        episode_row = episode_stats.result_set[0] if episode_stats.result_set else [0] * 10
        cue_row = cue_stats.result_set[0] if cue_stats.result_set else [0] * 11
        yield_row = yield_stats.result_set[0] if yield_stats.result_set else [0, 0]

        episode_count = episode_row[0] or 0
        projection_counts = {
            "queued": episode_row[1] or 0,
            "cued": episode_row[2] or 0,
            "cue_only": episode_row[3] or 0,
            "scheduled": episode_row[4] or 0,
            "projecting": episode_row[5] or 0,
            "projected": episode_row[6] or 0,
            "failed": episode_row[7] or 0,
            "dead_letter": episode_row[8] or 0,
        }
        cue_count = cue_row[0] or 0
        projected_cue_count = cue_row[10] or 0
        attempted_episode_count = (
            projection_counts["projected"]
            + projection_counts["failed"]
            + projection_counts["dead_letter"]
        )
        linked_entity_count = yield_row[0] or 0
        projected_relationship_count = yield_row[1] or 0

        cue_metrics = {
            "cue_count": cue_count,
            "episodes_without_cues": max(episode_count - cue_count, 0),
            "cue_coverage": round(cue_count / episode_count, 4) if episode_count else 0.0,
            "cue_hit_count": cue_row[2] or 0,
            "cue_hit_episode_count": cue_row[1] or 0,
            "cue_hit_episode_rate": round((cue_row[1] or 0) / cue_count, 4) if cue_count else 0.0,
            "cue_surfaced_count": cue_row[3] or 0,
            "cue_selected_count": cue_row[4] or 0,
            "cue_used_count": cue_row[5] or 0,
            "cue_near_miss_count": cue_row[6] or 0,
            "avg_policy_score": round(float(cue_row[7] or 0.0), 4),
            "avg_projection_attempts": round(float(cue_row[8] or 0.0), 4),
            "projected_cue_count": projected_cue_count,
            "cue_to_projection_conversion_rate": (
                round(projected_cue_count / cue_count, 4) if cue_count else 0.0
            ),
        }
        projection_metrics = {
            "state_counts": projection_counts,
            "attempted_episode_count": attempted_episode_count,
            "total_attempts": cue_row[9] or 0,
            "failure_count": projection_counts["failed"],
            "dead_letter_count": projection_counts["dead_letter"],
            "failure_rate": (
                round(
                    (projection_counts["failed"] + projection_counts["dead_letter"])
                    / attempted_episode_count,
                    4,
                )
                if attempted_episode_count
                else 0.0
            ),
            "avg_processing_duration_ms": round(float(episode_row[9] or 0.0), 2),
            "avg_time_to_projection_ms": 0.0,
            "yield": {
                "linked_entity_count": linked_entity_count,
                "relationship_count": projected_relationship_count,
                "avg_linked_entities_per_projected_episode": (
                    round(linked_entity_count / projection_counts["projected"], 4)
                    if projection_counts["projected"]
                    else 0.0
                ),
                "avg_relationships_per_projected_episode": (
                    round(
                        projected_relationship_count / projection_counts["projected"],
                        4,
                    )
                    if projection_counts["projected"]
                    else 0.0
                ),
            },
        }

        return {
            "entities": entity_count,
            "relationships": relationship_count,
            "episodes": episode_count,
            "cue_metrics": cue_metrics,
            "projection_metrics": projection_metrics,
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
        params: dict = {"limit": limit, "now": utc_now_iso()}
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

        since = (utc_now() - timedelta(days=days)).isoformat()
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
        params: dict = {
            "gid": group_id,
            "min_co": min_co_occurrence,
            "limit": limit,
            "now": utc_now_iso(),
        }
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

    async def find_structural_merge_candidates(
        self,
        group_id: str,
        min_shared_neighbors: int = 3,
        limit: int = 200,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs that share many active neighbors."""
        result = await self._query(
            """MATCH (a:Entity)-[r1:RELATES_TO]-(n:Entity)-[r2:RELATES_TO]-(b:Entity)
               WHERE a.group_id = $gid AND b.group_id = $gid AND n.group_id = $gid
                 AND a.deleted_at IS NULL AND b.deleted_at IS NULL AND n.deleted_at IS NULL
                 AND a.id < b.id
                 AND a.id <> b.id
                 AND (r1.valid_to IS NULL OR r1.valid_to > $now)
                 AND (r2.valid_to IS NULL OR r2.valid_to > $now)
               WITH a, b, COUNT(DISTINCT n) AS shared_count
               WHERE shared_count >= $min_shared_neighbors
               RETURN a.id, b.id, shared_count
               ORDER BY shared_count DESC
               LIMIT $limit""",
            {
                "gid": group_id,
                "min_shared_neighbors": min_shared_neighbors,
                "limit": limit,
                "now": utc_now_iso(),
            },
        )
        return [(row[0], row[1], row[2]) for row in result.result_set]

    async def get_episode_cooccurrence_count(
        self,
        entity_id_a: str,
        entity_id_b: str,
        group_id: str,
    ) -> int:
        """Count episodes where both entities are linked."""
        result = await self._query(
            """MATCH (ep:Episode)-[:HAS_ENTITY]->(a:Entity {id: $a}),
                     (ep)-[:HAS_ENTITY]->(b:Entity {id: $b})
               WHERE ep.group_id = $gid
               RETURN COUNT(DISTINCT ep)""",
            {"a": entity_id_a, "b": entity_id_b, "gid": group_id},
        )
        if not result.result_set:
            return 0
        return result.result_set[0][0] or 0

    async def get_dead_entities(
        self,
        group_id: str,
        min_age_days: int = 30,
        limit: int = 100,
        max_access_count: int = 0,
    ) -> list[Entity]:
        """Find entities with no relationships and low access."""
        cutoff = (utc_now() - timedelta(days=min_age_days)).isoformat()
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
                "gid": group_id,
                "cutoff": cutoff,
                "max_access": max_access_count,
                "limit": limit,
                "now": utc_now_iso(),
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

    async def get_entity_episode_count(self, entity_id: str, group_id: str) -> int:
        """Count episodes that mention this entity."""
        result = await self._query(
            """MATCH (ep:Episode)-[:HAS_ENTITY]->(e:Entity {id: $id})
               WHERE ep.group_id = $gid
               RETURN COUNT(DISTINCT ep)""",
            {"id": entity_id, "gid": group_id},
        )
        if not result.result_set:
            return 0
        return result.result_set[0][0] or 0

    async def get_entity_temporal_span(
        self,
        entity_id: str,
        group_id: str,
    ) -> tuple[str | None, str | None]:
        """Return the first and last episode timestamps mentioning this entity."""
        result = await self._query(
            """MATCH (ep:Episode)-[:HAS_ENTITY]->(e:Entity {id: $id})
               WHERE ep.group_id = $gid
               RETURN MIN(ep.created_at), MAX(ep.created_at)""",
            {"id": entity_id, "gid": group_id},
        )
        if not result.result_set:
            return (None, None)
        row = result.result_set[0]
        return (row[0], row[1])

    async def get_entity_relationship_types(
        self,
        entity_id: str,
        group_id: str,
    ) -> list[str]:
        """Return distinct active predicates connected to this entity."""
        result = await self._query(
            """MATCH (e:Entity {id: $id})-[r:RELATES_TO]-()
               WHERE r.group_id = $gid
                 AND (r.valid_to IS NULL OR r.valid_to > $now)
               RETURN DISTINCT r.predicate""",
            {"id": entity_id, "gid": group_id, "now": utc_now_iso()},
        )
        return [row[0] for row in result.result_set]

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
                keep_id,
                target_id,
                old_rel.predicate,
                group_id,
            )
            if existing:
                # Update weight if new is higher
                if old_rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        keep_id,
                        target_id,
                        old_rel.weight - existing.weight,
                        group_id=group_id,
                        predicate=old_rel.predicate,
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
                           source_episode: $source_episode, group_id: $gid,
                           polarity: $polarity
                       }]->(t)""",
                    {
                        "keep_id": keep_id,
                        "tgt_id": target_id,
                        "new_id": new_id,
                        "predicate": old_rel.predicate,
                        "weight": old_rel.weight,
                        "valid_from": (
                            old_rel.valid_from.isoformat() if old_rel.valid_from else None
                        ),
                        "valid_to": (old_rel.valid_to.isoformat() if old_rel.valid_to else None),
                        "created_at": (
                            old_rel.created_at.isoformat() if old_rel.created_at else None
                        ),
                        "confidence": old_rel.confidence,
                        "source_episode": old_rel.source_episode,
                        "gid": group_id,
                        "polarity": old_rel.polarity,
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
                source_id,
                keep_id,
                old_rel.predicate,
                group_id,
            )
            if existing:
                if old_rel.weight > existing.weight:
                    await self.update_relationship_weight(
                        source_id,
                        keep_id,
                        old_rel.weight - existing.weight,
                        group_id=group_id,
                        predicate=old_rel.predicate,
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
                           source_episode: $source_episode, group_id: $gid,
                           polarity: $polarity
                       }]->(k)""",
                    {
                        "src_id": source_id,
                        "keep_id": keep_id,
                        "new_id": new_id,
                        "predicate": old_rel.predicate,
                        "weight": old_rel.weight,
                        "valid_from": (
                            old_rel.valid_from.isoformat() if old_rel.valid_from else None
                        ),
                        "valid_to": (old_rel.valid_to.isoformat() if old_rel.valid_to else None),
                        "created_at": (
                            old_rel.created_at.isoformat() if old_rel.created_at else None
                        ),
                        "confidence": old_rel.confidence,
                        "source_episode": old_rel.source_episode,
                        "gid": group_id,
                        "polarity": old_rel.polarity,
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
                        keep_id,
                        remove_summary[:80],
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
                    "now": utc_now_iso(),
                },
            )

        # 5. Soft-delete loser
        await self._query(
            """MATCH (n:Entity {id: $id, group_id: $gid})
               SET n.deleted_at = $now, n.merged_into = $keep_id""",
            {
                "id": remove_id,
                "gid": group_id,
                "keep_id": keep_id,
                "now": utc_now_iso(),
            },
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
        now = utc_now_iso()
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
        now = utc_now_iso()
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

    async def sample_edges(
        self,
        group_id: str,
        limit: int = 500,
        exclude_ids: set[str] | None = None,
    ) -> list[Relationship]:
        """Return a random sample of active relationships."""
        now = utc_now_iso()
        exclude_clause = ""
        params: dict = {"gid": group_id, "now": now, "limit": limit}
        if exclude_ids:
            exclude_clause = " AND NOT r.id IN $excludes"
            params["excludes"] = list(exclude_ids)
        result = await self._query(
            f"""MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE r.group_id = $gid
                  AND (r.valid_to IS NULL OR r.valid_to > $now)
                  {exclude_clause}
                RETURN r
                ORDER BY rand()
                LIMIT $limit""",
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
            {"gid": group_id, "pred": predicate, "limit": limit, "now": utc_now_iso()},
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
            created_at=_parse_dt(props.get("created_at")) or utc_now(),
            updated_at=_parse_dt(props.get("updated_at")) or utc_now(),
            deleted_at=_parse_dt(props.get("deleted_at")),
            activation_current=props.get("activation_current", 0.0),
            access_count=props.get("access_count", 0),
            last_accessed=_parse_dt(props.get("last_accessed")),
            pii_detected=bool(props.get("pii_detected", False)),
            pii_categories=pii_categories,
            identity_core=bool(props.get("identity_core", False)),
            lexical_regime=props.get("lexical_regime"),
            canonical_identifier=props.get("canonical_identifier"),
            identifier_label=bool(props.get("identifier_label", False)),
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
            created_at=_parse_dt(props.get("created_at")) or utc_now(),
            confidence=props.get("confidence", 1.0),
            polarity=props.get("polarity", "positive"),
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
            conversation_date=_parse_dt(props.get("conversation_date")),
            created_at=_parse_dt(props.get("created_at")) or utc_now(),
            updated_at=_parse_dt(props.get("updated_at")),
            error=props.get("error"),
            retry_count=props.get("retry_count", 0) or 0,
            processing_duration_ms=props.get("processing_duration_ms"),
            encoding_context=props.get("encoding_context"),
            memory_tier=props.get("memory_tier", "episodic") or "episodic",
            consolidation_cycles=props.get("consolidation_cycles", 0) or 0,
            entity_coverage=props.get("entity_coverage", 0.0) or 0.0,
            projection_state=props.get("projection_state", "queued"),
            last_projection_reason=props.get("last_projection_reason"),
            last_projected_at=_parse_dt(props.get("last_projected_at")),
            attachments=[
                Attachment(**a)
                for a in json.loads(props.get("attachments_json", "[]") or "[]")
            ],
        )

    def _node_to_episode_cue(
        self,
        props: dict,
        episode_id: str,
        group_id: str,
    ) -> EpisodeCue:
        return EpisodeCue(
            episode_id=episode_id,
            group_id=group_id,
            cue_version=props.get("cue_version", 1),
            discourse_class=props.get("cue_discourse_class", "world"),
            projection_state=props.get("cue_projection_state", "cued"),
            cue_score=props.get("cue_score", 0.0) or 0.0,
            salience_score=props.get("cue_salience_score", 0.0) or 0.0,
            projection_priority=props.get("cue_projection_priority", 0.0) or 0.0,
            route_reason=props.get("cue_route_reason"),
            cue_text=props.get("cue_text", ""),
            entity_mentions=json.loads(props.get("cue_entity_mentions_json", "[]") or "[]"),
            temporal_markers=json.loads(props.get("cue_temporal_markers_json", "[]") or "[]"),
            quote_spans=json.loads(props.get("cue_quote_spans_json", "[]") or "[]"),
            contradiction_keys=json.loads(props.get("cue_contradiction_keys_json", "[]") or "[]"),
            first_spans=json.loads(props.get("cue_first_spans_json", "[]") or "[]"),
            hit_count=props.get("cue_hit_count", 0) or 0,
            surfaced_count=props.get("cue_surfaced_count", 0) or 0,
            selected_count=props.get("cue_selected_count", 0) or 0,
            used_count=props.get("cue_used_count", 0) or 0,
            near_miss_count=props.get("cue_near_miss_count", 0) or 0,
            policy_score=props.get("cue_policy_score", 0.0) or 0.0,
            projection_attempts=props.get("cue_projection_attempts", 0) or 0,
            last_hit_at=_parse_dt(props.get("cue_last_hit_at")),
            last_feedback_at=_parse_dt(props.get("cue_last_feedback_at")),
            last_projected_at=_parse_dt(props.get("cue_last_projected_at")),
            created_at=_parse_dt(props.get("cue_created_at")) or utc_now(),
            updated_at=_parse_dt(props.get("cue_updated_at")),
        )

    @staticmethod
    def _node_to_intention(node) -> Intention:
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
            created_at=_parse_dt(props.get("created_at")) or utc_now(),
            updated_at=_parse_dt(props.get("updated_at")) or utc_now(),
            expires_at=_parse_dt(props.get("expires_at")),
        )

    # --- Schema Formation (Brain Architecture Phase 3) stubs ---

    async def get_schema_members(
        self,
        schema_entity_id: str,
        group_id: str,
    ) -> list[dict]:
        """Stub — FalkorDB schema members not yet implemented."""
        return []

    async def save_schema_members(
        self,
        schema_entity_id: str,
        members: list[dict],
        group_id: str,
    ) -> None:
        """Stub — FalkorDB schema members not yet implemented."""

    async def find_entities_by_type(
        self,
        entity_type: str,
        group_id: str,
        limit: int = 100,
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
        i = cast("Intention", intention)
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
        self,
        group_id: str,
        enabled_only: bool = True,
    ) -> list:
        """List intentions for a group, filtering expired and optionally disabled."""
        if enabled_only:
            now = utc_now_iso()
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
        self,
        id: str,
        updates: dict,
        group_id: str,
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
        params["updated_at"] = utc_now_iso()
        cypher = f"MATCH (n:Intention {{id: $id, group_id: $gid}}) SET {', '.join(set_parts)}"
        await self._query(cypher, params)

    async def delete_intention(
        self,
        id: str,
        group_id: str,
        soft: bool = True,
    ) -> None:
        """Delete an intention (soft = disable, hard = remove)."""
        if soft:
            await self._query(
                """MATCH (n:Intention {id: $id, group_id: $gid})
                   SET n.enabled = false, n.updated_at = $now""",
                {"id": id, "gid": group_id, "now": utc_now_iso()},
            )
        else:
            await self._query(
                "MATCH (n:Intention {id: $id, group_id: $gid}) DELETE n",
                {"id": id, "gid": group_id},
            )

    async def increment_intention_fire_count(
        self,
        id: str,
        group_id: str,
    ) -> None:
        """Increment fire_count by 1."""
        await self._query(
            """MATCH (n:Intention {id: $id, group_id: $gid})
               SET n.fire_count = n.fire_count + 1, n.updated_at = $now""",
            {"id": id, "gid": group_id, "now": utc_now_iso()},
        )

    # --- Evidence storage (v2) ---

    async def store_evidence(
        self,
        evidence: list[dict],
        group_id: str = "default",
        *,
        default_status: str = "pending",
    ) -> None:
        """Persist evidence candidates as :Evidence nodes."""
        if not evidence:
            return
        for ev in evidence:
            status = ev.get("status", default_status)
            resolved_at = ev.get("resolved_at")
            if resolved_at is None and status in {
                "committed",
                "rejected",
                "expired",
                "superseded",
            }:
                resolved_at = utc_now_iso()
            await self._query(
                """MERGE (e:Evidence {evidence_id: $evidence_id, group_id: $group_id})
                   ON CREATE SET
                     e.episode_id = $episode_id,
                     e.fact_class = $fact_class,
                     e.confidence = $confidence,
                     e.source_type = $source_type,
                     e.extractor_name = $extractor_name,
                     e.payload_json = $payload_json,
                     e.source_span = $source_span,
                     e.signals_json = $signals_json,
                     e.ambiguity_tags_json = $ambiguity_tags_json,
                     e.ambiguity_score = $ambiguity_score,
                     e.adjudication_request_id = $adjudication_request_id,
                     e.status = $status,
                     e.commit_reason = $commit_reason,
                     e.committed_id = $committed_id,
                     e.deferred_cycles = $deferred_cycles,
                     e.created_at = $created_at,
                     e.resolved_at = $resolved_at""",
                {
                    "evidence_id": ev["evidence_id"],
                    "episode_id": ev["episode_id"],
                    "group_id": group_id,
                    "fact_class": ev["fact_class"],
                    "confidence": ev["confidence"],
                    "source_type": ev["source_type"],
                    "extractor_name": ev.get("extractor_name", ""),
                    "payload_json": json.dumps(ev.get("payload", {})),
                    "source_span": ev.get("source_span"),
                    "signals_json": json.dumps(ev.get("corroborating_signals", [])),
                    "ambiguity_tags_json": json.dumps(ev.get("ambiguity_tags", [])),
                    "ambiguity_score": ev.get("ambiguity_score", 0.0),
                    "adjudication_request_id": ev.get("adjudication_request_id"),
                    "status": status,
                    "commit_reason": ev.get("commit_reason"),
                    "committed_id": ev.get("committed_id"),
                    "deferred_cycles": ev.get("deferred_cycles", 0),
                    "created_at": ev.get("created_at", utc_now_iso()),
                    "resolved_at": resolved_at,
                },
            )

    async def get_pending_evidence(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Get unresolved evidence candidates for adjudication."""
        result = await self._query(
            """MATCH (e:Evidence {group_id: $gid})
               WHERE e.status IN ['pending', 'deferred', 'approved']
               RETURN e ORDER BY e.confidence DESC LIMIT $limit""",
            {"gid": group_id, "limit": limit},
        )
        return [_evidence_node_to_dict(row[0]) for row in result.result_set]

    async def get_episode_evidence(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Get all evidence associated with an episode."""
        result = await self._query(
            """MATCH (e:Evidence {episode_id: $episode_id, group_id: $gid})
               RETURN e ORDER BY e.confidence DESC""",
            {"episode_id": episode_id, "gid": group_id},
        )
        return [_evidence_node_to_dict(row[0]) for row in result.result_set]

    async def update_evidence_status(
        self,
        evidence_id: str,
        status: str,
        updates: dict | None = None,
        group_id: str = "default",
    ) -> None:
        """Update evidence status and optional metadata."""
        updates = updates or {}
        set_parts = ["e.status = $status"]
        params: dict = {
            "evidence_id": evidence_id,
            "gid": group_id,
            "status": status,
        }
        if "commit_reason" in updates:
            set_parts.append("e.commit_reason = $commit_reason")
            params["commit_reason"] = updates["commit_reason"]
        if "committed_id" in updates:
            set_parts.append("e.committed_id = $committed_id")
            params["committed_id"] = updates["committed_id"]
        if "confidence" in updates:
            set_parts.append("e.confidence = $confidence")
            params["confidence"] = updates["confidence"]
        if "deferred_cycles" in updates:
            set_parts.append("e.deferred_cycles = $deferred_cycles")
            params["deferred_cycles"] = updates["deferred_cycles"]
        if "ambiguity_tags" in updates:
            set_parts.append("e.ambiguity_tags_json = $ambiguity_tags_json")
            params["ambiguity_tags_json"] = json.dumps(updates["ambiguity_tags"])
        if "ambiguity_score" in updates:
            set_parts.append("e.ambiguity_score = $ambiguity_score")
            params["ambiguity_score"] = updates["ambiguity_score"]
        if "adjudication_request_id" in updates:
            set_parts.append("e.adjudication_request_id = $adjudication_request_id")
            params["adjudication_request_id"] = updates["adjudication_request_id"]
        if status in {"committed", "rejected", "expired", "superseded"}:
            set_parts.append("e.resolved_at = $resolved_at")
            params["resolved_at"] = utc_now_iso()
        await self._query(
            (
                "MATCH (e:Evidence {evidence_id: $evidence_id, group_id: $gid}) "
                f"SET {', '.join(set_parts)}"
            ),
            params,
        )

    async def get_entity_count(self, group_id: str = "default") -> int:
        """Count non-deleted entities in a group."""
        result = await self._query(
            """MATCH (n:Entity {group_id: $gid})
               WHERE n.deleted_at IS NULL
               RETURN COUNT(n)""",
            {"gid": group_id},
        )
        return result.result_set[0][0] if result.result_set else 0

    async def store_adjudication_requests(
        self,
        requests: list[dict],
        group_id: str = "default",
    ) -> None:
        """Persist edge adjudication requests."""
        for req in requests:
            await self._query(
                """MERGE (a:AdjudicationRequest {request_id: $request_id, group_id: $group_id})
                   ON CREATE SET
                     a.episode_id = $episode_id,
                     a.status = $status,
                     a.ambiguity_tags_json = $ambiguity_tags_json,
                     a.evidence_ids_json = $evidence_ids_json,
                     a.selected_text = $selected_text,
                     a.request_reason = $request_reason,
                     a.resolution_source = $resolution_source,
                     a.resolution_payload_json = $resolution_payload_json,
                     a.attempt_count = $attempt_count,
                     a.created_at = $created_at,
                     a.resolved_at = $resolved_at""",
                {
                    "request_id": req["request_id"],
                    "episode_id": req["episode_id"],
                    "group_id": group_id,
                    "status": req.get("status", "pending"),
                    "ambiguity_tags_json": json.dumps(req.get("ambiguity_tags", [])),
                    "evidence_ids_json": json.dumps(req.get("evidence_ids", [])),
                    "selected_text": req.get("selected_text", ""),
                    "request_reason": req.get("request_reason", ""),
                    "resolution_source": req.get("resolution_source"),
                    "resolution_payload_json": (
                        json.dumps(req.get("resolution_payload"))
                        if req.get("resolution_payload") is not None
                        else None
                    ),
                    "attempt_count": req.get("attempt_count", 0),
                    "created_at": req.get("created_at", utc_now_iso()),
                    "resolved_at": req.get("resolved_at"),
                },
            )

    async def get_episode_adjudications(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Get adjudication requests for an episode."""
        result = await self._query(
            """MATCH (a:AdjudicationRequest {episode_id: $episode_id, group_id: $gid})
               RETURN a ORDER BY a.created_at ASC""",
            {"episode_id": episode_id, "gid": group_id},
        )
        return [_adjudication_node_to_dict(row[0]) for row in result.result_set]

    async def get_adjudication_request(
        self,
        request_id: str,
        group_id: str = "default",
    ) -> dict | None:
        """Get a single adjudication request by ID."""
        result = await self._query(
            """MATCH (a:AdjudicationRequest {request_id: $request_id, group_id: $gid})
               RETURN a LIMIT 1""",
            {"request_id": request_id, "gid": group_id},
        )
        if not result.result_set:
            return None
        return _adjudication_node_to_dict(result.result_set[0][0])

    async def get_pending_adjudication_requests(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Get unresolved adjudication requests for consolidation."""
        result = await self._query(
            """MATCH (a:AdjudicationRequest {group_id: $gid})
               WHERE a.status IN ['pending', 'deferred', 'error']
               RETURN a ORDER BY a.created_at ASC LIMIT $limit""",
            {"gid": group_id, "limit": limit},
        )
        return [_adjudication_node_to_dict(row[0]) for row in result.result_set]

    async def update_adjudication_request(
        self,
        request_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        """Update adjudication request fields."""
        if not updates:
            return
        set_parts: list[str] = []
        params: dict = {"request_id": request_id, "gid": group_id}
        for field, prop in (
            ("status", "status"),
            ("selected_text", "selected_text"),
            ("request_reason", "request_reason"),
            ("resolution_source", "resolution_source"),
            ("attempt_count", "attempt_count"),
            ("resolved_at", "resolved_at"),
        ):
            if field not in updates:
                continue
            set_parts.append(f"a.{prop} = ${field}")
            params[field] = updates[field]
        if "ambiguity_tags" in updates:
            set_parts.append("a.ambiguity_tags_json = $ambiguity_tags_json")
            params["ambiguity_tags_json"] = json.dumps(updates["ambiguity_tags"])
        if "evidence_ids" in updates:
            set_parts.append("a.evidence_ids_json = $evidence_ids_json")
            params["evidence_ids_json"] = json.dumps(updates["evidence_ids"])
        if "resolution_payload" in updates:
            set_parts.append("a.resolution_payload_json = $resolution_payload_json")
            params["resolution_payload_json"] = (
                json.dumps(updates["resolution_payload"])
                if updates["resolution_payload"] is not None
                else None
            )
        status = updates.get("status")
        if status in {"materialized", "rejected", "expired"} and "resolved_at" not in updates:
            set_parts.append("a.resolved_at = $resolved_at_auto")
            params["resolved_at_auto"] = utc_now_iso()
        if not set_parts:
            return
        await self._query(
            (
                "MATCH (a:AdjudicationRequest {request_id: $request_id, group_id: $gid}) "
                f"SET {', '.join(set_parts)}"
            ),
            params,
        )
