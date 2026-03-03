"""Graph Manager — orchestrates extraction, entity resolution, and storage."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.conflicts import is_exclusive_predicate
from engram.extraction.extractor import EntityExtractor
from engram.extraction.resolver import resolve_entity
from engram.extraction.temporal import resolve_temporal_hint
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship
from engram.retrieval.pipeline import retrieve
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

logger = logging.getLogger(__name__)


class GraphManager:
    """Orchestrates: extract -> upsert entities -> create relationships -> index -> activate."""

    def __init__(
        self,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        extractor: EntityExtractor,
        cfg: ActivationConfig | None = None,
        event_bus: EventBus | None = None,
        canonicalizer: PredicateCanonicalizer | None = None,
        reranker: object | None = None,
        community_store: object | None = None,
        predicate_cache: object | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._extractor = extractor
        self._cfg = cfg or ActivationConfig()
        self._event_bus = event_bus
        self._canonicalizer = canonicalizer or PredicateCanonicalizer()
        self._reranker = reranker
        self._community_store = community_store
        self._predicate_cache = predicate_cache

        # Working memory buffer
        if self._cfg.working_memory_enabled:
            self._working_memory: WorkingMemoryBuffer | None = WorkingMemoryBuffer(
                capacity=self._cfg.working_memory_capacity,
                ttl_seconds=self._cfg.working_memory_ttl_seconds,
            )
        else:
            self._working_memory = None

    @staticmethod
    def _merge_entity_attributes(
        existing: Entity, new_summary: str | None, new_pii: bool = False,
        new_pii_categories: list[str] | None = None,
    ) -> dict:
        """Merge new attributes into an existing entity. Returns update dict."""
        updates: dict = {}

        # Append summaries (capped at 500 chars)
        if new_summary and new_summary != existing.summary:
            if existing.summary:
                merged = f"{existing.summary}; {new_summary}"
                if len(merged) > 500:
                    merged = merged[:497] + "..."
                updates["summary"] = merged
            else:
                updates["summary"] = new_summary[:500]

        # PII: once flagged, always flagged
        if new_pii and not existing.pii_detected:
            updates["pii_detected"] = 1

        if new_pii_categories:
            existing_cats = existing.pii_categories or []
            merged_cats = list(set(existing_cats + new_pii_categories))
            if merged_cats != existing_cats:
                import json
                updates["pii_categories"] = json.dumps(merged_cats)

        return updates

    def _publish(self, group_id: str, event_type: str, payload: dict | None = None) -> None:
        """Publish event if bus is configured."""
        if self._event_bus:
            self._event_bus.publish(group_id, event_type, payload)

    async def _update_episode_status(
        self, episode_id: str, status: EpisodeStatus, group_id: str = "default", **extra: object
    ) -> None:
        """Update episode status and updated_at timestamp."""
        updates: dict = {"status": status.value}
        updates.update(extra)
        await self._graph.update_episode(episode_id, updates, group_id=group_id)

    async def ingest_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Ingest a text episode: store, extract, resolve, link.

        Returns the episode ID. In lite mode this is synchronous.
        """
        start_ms = time.monotonic()

        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = Episode(
            id=episode_id,
            content=content,
            source=source,
            status=EpisodeStatus.QUEUED,
            group_id=group_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
        )
        await self._graph.create_episode(episode)
        self._publish(group_id, "episode.queued", {
            "episode": {
                "episodeId": episode_id,
                "content": content[:200] if content else "",
                "source": source or "unknown",
                "status": "queued",
                "createdAt": episode.created_at.isoformat() if episode.created_at else "",
                "updatedAt": episode.created_at.isoformat() if episode.created_at else "",
                "entities": [],
                "factsCount": 0,
                "processingDurationMs": None,
                "error": None,
                "retryCount": 0,
            },
        })

        try:
            # Extraction
            await self._update_episode_status(
                episode_id, EpisodeStatus.EXTRACTING, group_id=group_id,
            )
            result = await self._extractor.extract(content)

            # Resolution
            await self._update_episode_status(
                episode_id, EpisodeStatus.RESOLVING, group_id=group_id,
            )
            existing = await self._graph.find_entities(group_id=group_id, limit=1000)
            entity_map: dict[str, str] = {}  # extracted_name -> entity_id

            now = time.time()
            new_entity_names: list[str] = []

            for ent_data in result.entities:
                name = ent_data["name"]
                entity_type = ent_data.get("entity_type", "Other")
                summary = ent_data.get("summary")
                pii_detected = ent_data.get("pii_detected", False)
                pii_categories = ent_data.get("pii_categories")

                existing_entity = await resolve_entity(name, entity_type, existing)

                if existing_entity:
                    entity_id = existing_entity.id
                    updates = self._merge_entity_attributes(
                        existing_entity, summary, pii_detected, pii_categories
                    )
                    if updates:
                        await self._graph.update_entity(entity_id, updates, group_id=group_id)
                else:
                    entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                    entity = Entity(
                        id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        summary=summary,
                        group_id=group_id,
                        pii_detected=pii_detected,
                        pii_categories=pii_categories,
                    )
                    await self._graph.create_entity(entity)
                    existing.append(entity)
                    new_entity_names.append(name)

                entity_map[name] = entity_id
                await self._graph.link_episode_entity(episode_id, entity_id)

                # Record access for extracted entity
                await self._activation.record_access(entity_id, now, group_id=group_id)

            # Writing relationships
            await self._update_episode_status(episode_id, EpisodeStatus.WRITING, group_id=group_id)
            for rel_data in result.relationships:
                source_name = rel_data.get("source") or rel_data.get("source_entity", "")
                target_name = rel_data.get("target") or rel_data.get("target_entity", "")
                source_id = entity_map.get(source_name)
                target_id = entity_map.get(target_name)

                if source_id and target_id:
                    predicate = (
                        (
                            rel_data.get("predicate")
                            or rel_data.get("relationship_type")
                            or rel_data.get("type")
                            or "RELATES_TO"
                        )
                        .upper()
                        .replace(" ", "_")
                    )
                    predicate = self._canonicalizer.canonicalize(predicate)

                    # Parse temporal fields
                    valid_from_str = rel_data.get("valid_from")
                    valid_to_str = rel_data.get("valid_to")
                    temporal_hint = rel_data.get("temporal_hint")

                    dt_now = datetime.utcnow()
                    valid_from = None
                    valid_to = None
                    confidence = 1.0

                    # Resolve valid_from
                    if valid_from_str:
                        try:
                            valid_from = datetime.fromisoformat(valid_from_str)
                            confidence = 1.0
                        except (ValueError, TypeError):
                            resolved = resolve_temporal_hint(valid_from_str, dt_now)
                            if resolved:
                                valid_from = resolved
                                confidence = 0.8
                    elif temporal_hint:
                        resolved = resolve_temporal_hint(temporal_hint, dt_now)
                        if resolved:
                            valid_from = resolved
                            confidence = 0.8
                        else:
                            confidence = 0.7

                    # Resolve valid_to
                    if valid_to_str:
                        try:
                            valid_to = datetime.fromisoformat(valid_to_str)
                        except (ValueError, TypeError):
                            resolved = resolve_temporal_hint(valid_to_str, dt_now)
                            if resolved:
                                valid_to = resolved

                    if valid_from is None:
                        valid_from = dt_now

                    # Conflict detection for exclusive predicates
                    if is_exclusive_predicate(predicate):
                        conflicts = await self._graph.find_conflicting_relationships(
                            source_id, predicate, group_id
                        )
                        for conflict in conflicts:
                            # Don't invalidate if same target (not a real conflict)
                            if conflict.target_id == target_id:
                                continue
                            await self._graph.invalidate_relationship(
                                conflict.id, valid_from, group_id=group_id
                            )
                            logger.info(
                                "Invalidated conflicting relationship %s "
                                "(%s → %s via %s)",
                                conflict.id,
                                conflict.source_id,
                                conflict.target_id,
                                conflict.predicate,
                            )

                    rel = Relationship(
                        id=f"rel_{uuid.uuid4().hex[:12]}",
                        source_id=source_id,
                        target_id=target_id,
                        predicate=predicate,
                        weight=float(rel_data.get("weight", 1.0)),
                        valid_from=valid_from,
                        valid_to=valid_to,
                        confidence=confidence,
                        source_episode=episode_id,
                        group_id=group_id,
                    )
                    await self._graph.create_relationship(rel)

            # Publish graph changes
            self._publish(group_id, "graph.nodes_added", {
                "episode_id": episode_id,
                "entity_count": len(result.entities),
                "relationship_count": len(result.relationships),
                "new_entities": new_entity_names,
            })

            # ── Embedding ──
            await self._update_episode_status(
                episode_id, EpisodeStatus.EMBEDDING, group_id=group_id,
            )
            try:
                # Index changed entities for vector search
                for ent_data in result.entities:
                    eid = entity_map.get(ent_data["name"])
                    if eid:
                        ent = await self._graph.get_entity(eid, group_id)
                        if ent:
                            if self._cfg.structure_aware_embeddings:
                                await self._index_entity_with_structure(
                                    ent, group_id,
                                )
                            else:
                                await self._search.index_entity(ent)
                # Index episode content
                ep = await self._graph.get_episode_by_id(episode_id, group_id)
                if ep:
                    await self._search.index_episode(ep)
            except Exception as embed_err:
                logger.warning(
                    "Embedding failed for episode %s (non-fatal): %s",
                    episode_id, embed_err,
                )
                # Continue — FTS5 still works

            # ── Activating ──
            await self._update_episode_status(
                episode_id, EpisodeStatus.ACTIVATING, group_id=group_id,
            )
            # Activation recording already happens during entity extraction above

            # Complete
            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            await self._update_episode_status(
                episode_id, EpisodeStatus.COMPLETED,
                group_id=group_id,
                processing_duration_ms=elapsed_ms,
            )
            self._publish(group_id, "episode.completed", {
                "episodeId": episode_id,
                "status": "completed",
                "entity_count": len(result.entities),
                "relationship_count": len(result.relationships),
                "duration_ms": elapsed_ms,
            })

            logger.info(
                "Ingested episode %s: %d entities, %d relationships",
                episode_id,
                len(result.entities),
                len(result.relationships),
            )

        except Exception as e:
            logger.error("Failed to process episode %s: %s", episode_id, e)
            await self._update_episode_status(
                episode_id, EpisodeStatus.FAILED, group_id=group_id, error=str(e),
            )
            self._publish(group_id, "episode.failed", {
                "episodeId": episode_id,
                "status": "failed",
                "error": str(e),
            })

        return episode_id

    async def _index_entity_with_structure(
        self, entity: Entity, group_id: str,
    ) -> None:
        """Build structure-aware embedding text that includes relationship predicates."""
        rels = await self._graph.get_relationships(
            entity.id, active_only=True, group_id=group_id,
        )

        # Sort by predicate weight (high-signal predicates first)
        predicate_weights = self._cfg.predicate_weights
        default_weight = self._cfg.predicate_weight_default
        rels_sorted = sorted(
            rels,
            key=lambda r: predicate_weights.get(r.predicate, default_weight),
            reverse=True,
        )

        natural_names = self._cfg.predicate_natural_names
        max_rels = self._cfg.structure_max_relationships
        rel_parts: list[str] = []
        for r in rels_sorted[:max_rels]:
            pred_natural = natural_names.get(
                r.predicate, r.predicate.lower().replace("_", " ")
            )
            if r.source_id == entity.id:
                target = await self._graph.get_entity(r.target_id, group_id)
                target_name = target.name if target else r.target_id
                rel_parts.append(f"{entity.name} {pred_natural} {target_name}")
            else:
                source = await self._graph.get_entity(r.source_id, group_id)
                source_name = source.name if source else r.source_id
                rel_parts.append(f"{source_name} {pred_natural} {entity.name}")

        # Build enriched text: "{name}. {type}. {summary}. Relationships: ..."
        parts = [entity.name]
        if entity.entity_type:
            parts.append(entity.entity_type)
        if entity.summary:
            parts.append(entity.summary)

        text = ". ".join(parts) + "."
        if rel_parts:
            text += " Relationships: " + ", ".join(rel_parts)

        enriched = Entity(
            id=entity.id,
            name=text,
            entity_type=entity.entity_type,
            summary=None,
            group_id=entity.group_id,
        )
        await self._search.index_entity(enriched)

    async def recall(
        self,
        query: str,
        group_id: str = "default",
        limit: int = 10,
    ) -> list[dict]:
        """Retrieve relevant entities and their context using activation-aware scoring."""
        scored_results = await retrieve(
            query=query,
            group_id=group_id,
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            limit=limit,
            working_memory=self._working_memory,
            reranker=self._reranker,
            community_store=self._community_store,
            predicate_cache=self._predicate_cache,
        )

        now = time.time()
        results = []
        for sr in scored_results:
            if sr.result_type == "episode":
                # Fetch episode data — do NOT record access for episodes
                ep = await self._graph.get_episode_by_id(sr.node_id, group_id)
                if ep:
                    linked_entities = await self._graph.get_episode_entities(sr.node_id)

                    # Populate working memory buffer for episodes
                    if self._working_memory is not None:
                        self._working_memory.add(
                            sr.node_id, "episode", sr.score, query, now,
                        )

                    results.append(
                        {
                            "episode": {
                                "id": ep.id,
                                "content": ep.content[:500],
                                "source": ep.source,
                                "created_at": ep.created_at.isoformat() if ep.created_at else None,
                            },
                            "score": sr.score,
                            "score_breakdown": {
                                "semantic": sr.semantic_similarity,
                                "activation": sr.activation,
                                "edge_proximity": sr.edge_proximity,
                                "exploration_bonus": sr.exploration_bonus,
                            },
                            "result_type": "episode",
                            "linked_entities": linked_entities,
                        }
                    )
            else:
                entity = await self._graph.get_entity(sr.node_id, group_id)
                if entity:
                    rels = await self._graph.get_relationships(sr.node_id, group_id=group_id)

                    # Record access for returned entities
                    await self._activation.record_access(sr.node_id, now, group_id=group_id)

                    # Populate working memory buffer for entities
                    if self._working_memory is not None:
                        self._working_memory.add(
                            sr.node_id, "entity", sr.score, query, now,
                        )

                    results.append(
                        {
                            "entity": {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.entity_type,
                                "summary": entity.summary,
                            },
                            "score": sr.score,
                            "score_breakdown": {
                                "semantic": sr.semantic_similarity,
                                "activation": sr.activation,
                                "edge_proximity": sr.edge_proximity,
                                "exploration_bonus": sr.exploration_bonus,
                            },
                            "relationships": [
                                {
                                    "predicate": r.predicate,
                                    "source_id": r.source_id,
                                    "target_id": r.target_id,
                                    "weight": r.weight,
                                }
                                for r in rels[:5]
                            ],
                        }
                    )

        # Track the query in working memory
        if self._working_memory is not None:
            self._working_memory.add_query(query, now)

        return results

    # ─── Entity name resolution ─────────────────────────────────────

    async def resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Resolve an entity ID to its name. Returns ID if not found."""
        entity = await self._graph.get_entity(entity_id, group_id)
        return entity.name if entity else entity_id

    # ─── Search entities ────────────────────────────────────────────

    async def search_entities(
        self,
        group_id: str = "default",
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search entities by name (FTS5 fuzzy) and/or type. Read-only, does not record access."""
        from engram.activation.engine import compute_activation

        now = time.time()
        entities: list[Entity] = []

        if name:
            # Use FTS5 for fuzzy matching, then optionally filter by type
            search_hits = await self._search.search(
                query=name, group_id=group_id, limit=limit * 2
            )
            for eid, _score in search_hits:
                ent = await self._graph.get_entity(eid, group_id)
                if ent and (not entity_type or ent.entity_type == entity_type):
                    entities.append(ent)
                if len(entities) >= limit:
                    break
        else:
            # Type-only search
            entities = await self._graph.find_entities(
                entity_type=entity_type, group_id=group_id, limit=limit
            )

        result = []
        entity_ids = [e.id for e in entities]
        states = await self._activation.batch_get(entity_ids)

        for entity in entities:
            state = states.get(entity.id)
            activation_score = 0.0
            access_count = 0
            if state:
                activation_score = compute_activation(state.access_history, now, self._cfg)
                access_count = state.access_count

            result.append({
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "summary": entity.summary,
                "activation_score": round(activation_score, 4),
                "created_at": entity.created_at.isoformat() if entity.created_at else None,
                "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                "access_count": access_count,
            })

        return result

    # ─── Search facts ───────────────────────────────────────────────

    async def search_facts(
        self,
        group_id: str = "default",
        query: str = "",
        subject: str | None = None,
        predicate: str | None = None,
        include_expired: bool = False,
        limit: int = 10,
    ) -> list[dict]:
        """Search for relationships/facts. Resolves entity names."""
        # Normalize predicate filter
        if predicate:
            predicate = predicate.upper().replace(" ", "_")

        relationships: list[Relationship] = []

        if subject:
            # Resolve subject name to entity ID
            subject_entities = await self._graph.find_entities(
                name=subject, group_id=group_id, limit=1
            )
            if not subject_entities:
                # Try FTS5 as fallback
                hits = await self._search.search(query=subject, group_id=group_id, limit=5)
                for eid, _ in hits:
                    ent = await self._graph.get_entity(eid, group_id)
                    if ent and ent.name.lower() == subject.lower():
                        subject_entities = [ent]
                        break

            if subject_entities:
                subject_id = subject_entities[0].id
                rels = await self._graph.get_relationships(
                    subject_id,
                    direction="outgoing",
                    predicate=predicate,
                    active_only=not include_expired,
                    group_id=group_id,
                )
                relationships.extend(rels)
        else:
            # FTS5 search, get relationships for top hits
            search_hits = await self._search.search(
                query=query, group_id=group_id, limit=limit
            )
            seen_rel_ids: set[str] = set()
            for eid, _ in search_hits:
                rels = await self._graph.get_relationships(
                    eid,
                    direction="both",
                    predicate=predicate,
                    active_only=not include_expired,
                    group_id=group_id,
                )
                for r in rels:
                    if r.id not in seen_rel_ids:
                        seen_rel_ids.add(r.id)
                        relationships.append(r)
                if len(relationships) >= limit:
                    break

        # Resolve entity names and format results
        result = []
        for r in relationships[:limit]:
            source_name = await self.resolve_entity_name(r.source_id, group_id)
            target_name = await self.resolve_entity_name(r.target_id, group_id)
            result.append({
                "subject": source_name,
                "predicate": r.predicate,
                "object": target_name,
                "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                "confidence": r.confidence,
                "source_episode": r.source_episode,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        return result

    # ─── Forget entity ──────────────────────────────────────────────

    async def forget_entity(
        self,
        entity_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Soft-delete an entity and clear its activation."""
        entities = await self._graph.find_entities(
            name=entity_name, group_id=group_id, limit=1
        )
        if not entities:
            return {"status": "error", "message": f"Entity '{entity_name}' not found."}

        entity = entities[0]
        await self._graph.delete_entity(entity.id, soft=True, group_id=group_id)
        await self._activation.clear_activation(entity.id)

        logger.info("Forgot entity %s (%s), reason: %s", entity.name, entity.id, reason)
        return {
            "status": "forgotten",
            "target_type": "entity",
            "target": entity.name,
            "valid_to": datetime.utcnow().isoformat(),
            "message": f"Entity '{entity.name}' has been forgotten.",
        }

    # ─── Forget fact ────────────────────────────────────────────────

    async def forget_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Invalidate a specific relationship (fact)."""
        predicate = predicate.upper().replace(" ", "_")

        # Resolve entity names to IDs
        subject_entities = await self._graph.find_entities(
            name=subject_name, group_id=group_id, limit=1
        )
        object_entities = await self._graph.find_entities(
            name=object_name, group_id=group_id, limit=1
        )

        if not subject_entities:
            return {"status": "error", "message": f"Subject '{subject_name}' not found."}
        if not object_entities:
            return {"status": "error", "message": f"Object '{object_name}' not found."}

        subject_id = subject_entities[0].id
        object_id = object_entities[0].id

        rels = await self._graph.get_relationships(
            subject_id, direction="outgoing", predicate=predicate, active_only=True,
            group_id=group_id,
        )
        target_rel = None
        for r in rels:
            if r.target_id == object_id:
                target_rel = r
                break

        if not target_rel:
            return {
                "status": "error",
                "message": (
                    f"No active fact found: {subject_name} —{predicate}→ {object_name}."
                ),
            }

        await self._graph.invalidate_relationship(
            target_rel.id, datetime.utcnow(), group_id=group_id
        )

        logger.info(
            "Forgot fact %s —%s→ %s, reason: %s",
            subject_name, predicate, object_name, reason,
        )
        return {
            "status": "forgotten",
            "target_type": "fact",
            "subject": subject_name,
            "predicate": predicate,
            "object": object_name,
            "valid_to": datetime.utcnow().isoformat(),
            "message": f"Fact '{subject_name} {predicate} {object_name}' has been forgotten.",
        }

    # ─── Get context ────────────────────────────────────────────────

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
    ) -> dict:
        """Build a markdown context summary of the most activated memories."""
        from engram.activation.engine import compute_activation

        now = time.time()
        entities_data = []
        facts_data = []

        if topic_hint:
            # Topic-biased: use recall to find relevant entities
            results = await self.recall(query=topic_hint, group_id=group_id, limit=20)
            for r in results:
                if r.get("result_type") == "episode":
                    continue
                ent = r["entity"]
                state = await self._activation.get_activation(ent["id"])
                act = 0.0
                if state:
                    act = compute_activation(state.access_history, now, self._cfg)
                entities_data.append({
                    "name": ent["name"],
                    "type": ent["type"],
                    "summary": ent.get("summary") or "",
                    "activation": act,
                    "id": ent["id"],
                })
                for rel in r.get("relationships", []):
                    source_name = await self.resolve_entity_name(rel["source_id"], group_id)
                    target_name = await self.resolve_entity_name(rel["target_id"], group_id)
                    facts_data.append(f"{source_name} {rel['predicate']} {target_name}")
        else:
            # Broad: get top activated entities
            top = await self._activation.get_top_activated(group_id=group_id, limit=20)
            for eid, state in top:
                entity = await self._graph.get_entity(eid, group_id)
                if entity:
                    act = compute_activation(state.access_history, now, self._cfg)
                    entities_data.append({
                        "name": entity.name,
                        "type": entity.entity_type,
                        "summary": entity.summary or "",
                        "activation": act,
                        "id": entity.id,
                    })
                    rels = await self._graph.get_relationships(
                        eid, active_only=True, group_id=group_id
                    )
                    for r in rels[:3]:
                        source_name = await self.resolve_entity_name(r.source_id, group_id)
                        target_name = await self.resolve_entity_name(r.target_id, group_id)
                        fact = f"{source_name} {r.predicate} {target_name}"
                        if fact not in facts_data:
                            facts_data.append(fact)

        # Sort by activation (highest first)
        entities_data.sort(key=lambda x: x["activation"], reverse=True)

        # Build markdown
        lines = ["## Active Memory Context", ""]
        lines.append("**Key entities (by activation):**")
        for ed in entities_data:
            summary_part = f" — {ed['summary']}" if ed["summary"] else ""
            lines.append(
                f"- **{ed['name']}** ({ed['type']}, activation={ed['activation']:.2f})"
                f"{summary_part}"
            )

        if facts_data:
            lines.append("")
            lines.append("**Recent facts:**")
            # Deduplicate
            seen = set()
            for fact in facts_data:
                if fact not in seen:
                    seen.add(fact)
                    lines.append(f"- {fact}")

        # Active topics (unique entity types)
        types = list({ed["type"] for ed in entities_data})
        if types:
            lines.append("")
            lines.append(f"**Active topics:** {', '.join(sorted(types))}")

        context_text = "\n".join(lines)

        # Token estimate and truncation
        token_estimate = len(context_text) // 4
        if token_estimate > max_tokens:
            # Truncate to fit budget
            char_budget = max_tokens * 4
            context_text = context_text[:char_budget]
            token_estimate = max_tokens

        # Record access for included entities (readOnlyHint=false per spec)
        for ed in entities_data:
            await self._activation.record_access(ed["id"], now, group_id=group_id)

        return {
            "context": context_text,
            "entity_count": len(entities_data),
            "fact_count": len(facts_data),
            "token_estimate": token_estimate,
        }

    # ─── Get graph state ────────────────────────────────────────────

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

        # Get top activated
        top = await self._activation.get_top_activated(group_id=group_id, limit=top_n * 2)

        top_activated = []
        active_count = 0
        dormant_count = 0

        for eid, state in top:
            entity = await self._graph.get_entity(eid, group_id)
            if not entity:
                continue
            if entity_types and entity.entity_type not in entity_types:
                continue

            act = compute_activation(state.access_history, now, self._cfg)
            if act > 0.3:
                active_count += 1
            else:
                dormant_count += 1

            if len(top_activated) < top_n:
                top_activated.append({
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "activation": round(act, 4),
                    "access_count": state.access_count,
                })

        stats["active_entities"] = active_count
        stats["dormant_entities"] = dormant_count

        result: dict = {
            "stats": stats,
            "top_activated": top_activated,
            "group_id": group_id,
        }

        if include_edges:
            edges = []
            seen_rel_ids: set[str] = set()
            for ta in top_activated:
                rels = await self._graph.get_relationships(
                    ta["id"], active_only=True, group_id=group_id
                )
                for r in rels:
                    if r.id not in seen_rel_ids:
                        seen_rel_ids.add(r.id)
                        source_name = await self.resolve_entity_name(r.source_id, group_id)
                        target_name = await self.resolve_entity_name(r.target_id, group_id)
                        edges.append({
                            "source": source_name,
                            "target": target_name,
                            "predicate": r.predicate,
                            "weight": r.weight,
                        })
            result["edges"] = edges

        return result
