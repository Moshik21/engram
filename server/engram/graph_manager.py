"""Graph Manager — orchestrates extraction, entity resolution, and storage."""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.conflicts import is_exclusive_predicate
from engram.extraction.discourse import classify_discourse
from engram.extraction.extractor import EntityExtractor
from engram.extraction.resolver import resolve_entity_fast
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

        # Briefing cache: (group_id, topic_hint) -> (timestamp, text)
        self._briefing_cache: dict[tuple[str, str | None], tuple[float, str]] = {}

        # Working memory buffer
        if self._cfg.working_memory_enabled:
            self._working_memory: WorkingMemoryBuffer | None = WorkingMemoryBuffer(
                capacity=self._cfg.working_memory_capacity,
                ttl_seconds=self._cfg.working_memory_ttl_seconds,
            )
        else:
            self._working_memory = None

    # Entity types where meta-summaries should be rejected (real-world entities)
    _PROTECTED_ENTITY_TYPES = frozenset(
        {"Person", "CreativeWork", "Location", "Event", "Organization",
         "Emotion", "Goal", "Preference"}
    )

    # Patterns that indicate a summary contains system-internal meta-commentary
    _META_SUMMARY_PATTERN = re.compile(
        r"activation[ _]?(?:score|current)|access[_ ]?count"
        r"|knowledge graph|graph (?:node|entity|store)"
        r"|retrieval|embedding|consolidation|triage"
        r"|entity (?:resolution|extraction|in the)"
        r"|cold session|test case|example case"
        r"|MCP tool|episode worker|spreading activation"
        r"|\b(?:ent|ep|rel|cyc)_[a-f0-9]",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_meta_summary(text: str) -> bool:
        """Check if a summary fragment contains system-internal patterns."""
        return bool(GraphManager._META_SUMMARY_PATTERN.search(text))

    @staticmethod
    def _merge_entity_attributes(
        existing: Entity,
        new_summary: str | None,
        new_pii: bool = False,
        new_pii_categories: list[str] | None = None,
        new_attributes: dict | None = None,
    ) -> dict:
        """Merge new attributes into an existing entity. Returns update dict."""
        updates: dict = {}

        # Append summaries (capped at 500 chars)
        if new_summary and new_summary != existing.summary:
            # Guard: reject meta-contaminated summaries for protected entity types
            entity_type = getattr(existing, "entity_type", "Other") or "Other"
            if (
                GraphManager._is_meta_summary(new_summary)
                and entity_type in GraphManager._PROTECTED_ENTITY_TYPES
            ):
                logger.warning(
                    "Rejected meta-summary for %s entity %r: %s",
                    entity_type,
                    existing.name,
                    new_summary[:80],
                )
            elif existing.summary:
                merged = f"{existing.summary}; {new_summary}"
                if len(merged) > 500:
                    merged = merged[:497] + "..."
                updates["summary"] = merged
            else:
                updates["summary"] = new_summary[:500]

        # Structured attributes: new values overwrite old for same key
        if new_attributes:
            merged_attrs = {**(existing.attributes or {}), **new_attributes}
            if merged_attrs != (existing.attributes or {}):
                import json

                updates["attributes"] = json.dumps(merged_attrs)

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

    async def _publish_access_event(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        """Compute activation and publish an activation.access event."""
        if not self._event_bus:
            return
        from engram.activation.engine import compute_activation

        now = time.time()
        state = await self._activation.get_activation(entity_id)
        activation = 0.0
        if state:
            activation = compute_activation(state.access_history, now, self._cfg)
        self._publish(
            group_id,
            "activation.access",
            {
                "entityId": entity_id,
                "name": name,
                "entityType": entity_type,
                "activation": round(activation, 4),
                "accessedVia": accessed_via,
            },
        )

    async def _update_episode_status(
        self, episode_id: str, status: EpisodeStatus, group_id: str = "default", **extra: object
    ) -> None:
        """Update episode status and updated_at timestamp."""
        updates: dict = {"status": status.value}
        updates.update(extra)
        await self._graph.update_episode(episode_id, updates, group_id=group_id)

    async def store_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Store a raw episode without extraction. Fast path for bulk capture.

        Returns the episode ID. The episode is created with QUEUED status.
        Call project_episode() later to run extraction.
        """
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
        self._publish(
            group_id,
            "episode.queued",
            {
                "episode": {
                    "episodeId": episode_id,
                    "content": content[:200] if content else "",
                    "source": source or "unknown",
                    "status": "queued",
                    "createdAt": (
                        episode.created_at.isoformat() + "Z"
                        if episode.created_at
                        else ""
                    ),
                    "updatedAt": (
                        episode.created_at.isoformat() + "Z"
                        if episode.created_at
                        else ""
                    ),
                    "entities": [],
                    "factsCount": 0,
                    "processingDurationMs": None,
                    "error": None,
                    "retryCount": 0,
                },
            },
        )
        return episode_id

    async def project_episode(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> None:
        """Run extraction, resolution, and embedding on a stored episode.

        Raises on failure after setting FAILED status.
        """
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        start_ms = time.monotonic()
        content = episode.content

        # Discourse gate: skip pure system meta-commentary
        if classify_discourse(content) == "system":
            logger.warning(
                "project_episode: skipping system-discourse episode %s", episode_id
            )
            await self._update_episode_status(
                episode_id, EpisodeStatus.COMPLETED, group_id=group_id,
                skipped_meta=True,
            )
            return

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
            session_entities: dict[str, Entity] = {}
            entity_map: dict[str, str] = {}

            async def _get_candidates(name: str, gid: str) -> list[Entity]:
                return await self._graph.find_entity_candidates(name, gid)

            now = time.time()
            new_entity_names: list[str] = []

            meta_entity_names: set[str] = set()

            for ent_data in result.entities:
                name = ent_data["name"]
                entity_type = ent_data.get("entity_type", "Other")
                summary = ent_data.get("summary")
                attributes = ent_data.get("attributes") or None
                pii_detected = ent_data.get("pii_detected", False)
                pii_categories = ent_data.get("pii_categories")

                # Fix 5: skip entities tagged as meta by LLM
                if ent_data.get("epistemic_mode") == "meta":
                    logger.debug(
                        "Skipping meta-tagged entity %r (type=%s)", name, entity_type
                    )
                    meta_entity_names.add(name)
                    continue

                existing_entity = await resolve_entity_fast(
                    name, entity_type, _get_candidates, group_id,
                    session_entities=session_entities,
                )

                if existing_entity:
                    entity_id = existing_entity.id
                    updates = self._merge_entity_attributes(
                        existing_entity, summary, pii_detected, pii_categories,
                        new_attributes=attributes,
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
                        attributes=attributes,
                        group_id=group_id,
                        pii_detected=pii_detected,
                        pii_categories=pii_categories,
                    )
                    await self._graph.create_entity(entity)
                    session_entities[entity_id] = entity
                    new_entity_names.append(name)

                entity_map[name] = entity_id
                await self._graph.link_episode_entity(episode_id, entity_id)
                await self._activation.record_access(entity_id, now, group_id=group_id)
                await self._publish_access_event(
                    entity_id, name, entity_type, group_id, "ingest"
                )

            # Writing relationships
            await self._update_episode_status(
                episode_id, EpisodeStatus.WRITING, group_id=group_id,
            )
            for rel_data in result.relationships:
                source_name = rel_data.get("source") or rel_data.get("source_entity", "")
                target_name = rel_data.get("target") or rel_data.get("target_entity", "")

                # Skip relationships involving meta-tagged entities
                if source_name in meta_entity_names or target_name in meta_entity_names:
                    logger.debug(
                        "Skipping relationship %s->%s (meta entity involved)",
                        source_name,
                        target_name,
                    )
                    continue

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

                    valid_from_str = rel_data.get("valid_from")
                    valid_to_str = rel_data.get("valid_to")
                    temporal_hint = rel_data.get("temporal_hint")

                    dt_now = datetime.utcnow()
                    valid_from = None
                    valid_to = None
                    confidence = 1.0

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

                    if valid_to_str:
                        try:
                            valid_to = datetime.fromisoformat(valid_to_str)
                        except (ValueError, TypeError):
                            resolved = resolve_temporal_hint(valid_to_str, dt_now)
                            if resolved:
                                valid_to = resolved

                    if valid_from is None:
                        valid_from = dt_now

                    # Polarity handling
                    polarity = rel_data.get("polarity", "positive")
                    if polarity not in ("positive", "negative", "uncertain"):
                        polarity = "positive"

                    rel_weight = float(rel_data.get("weight", 1.0))

                    if polarity == "negative":
                        # Invalidate existing positive relationships with same
                        # source+target+predicate
                        existing_rels = await self._graph.get_relationships(
                            source_id,
                            direction="outgoing",
                            predicate=predicate,
                            active_only=True,
                            group_id=group_id,
                        )
                        for existing_rel in existing_rels:
                            if existing_rel.target_id == target_id:
                                await self._graph.invalidate_relationship(
                                    existing_rel.id, dt_now, group_id=group_id
                                )
                                logger.info(
                                    "Negation invalidated relationship %s "
                                    "(%s → %s via %s)",
                                    existing_rel.id,
                                    existing_rel.source_id,
                                    existing_rel.target_id,
                                    existing_rel.predicate,
                                )
                    elif polarity == "uncertain":
                        rel_weight *= 0.5

                    if is_exclusive_predicate(predicate):
                        conflicts = await self._graph.find_conflicting_relationships(
                            source_id, predicate, group_id
                        )
                        for conflict in conflicts:
                            if conflict.target_id == target_id:
                                continue
                            await self._graph.invalidate_relationship(
                                conflict.id, valid_from, group_id=group_id
                            )
                            logger.info(
                                "Invalidated conflicting relationship %s (%s → %s via %s)",
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
                        weight=rel_weight,
                        polarity=polarity,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        confidence=confidence,
                        source_episode=episode_id,
                        group_id=group_id,
                    )
                    await self._graph.create_relationship(rel)

                    # Auto-detect identity core entities via identity predicates
                    if (
                        self._cfg.identity_core_enabled
                        and predicate in self._cfg.identity_predicates
                    ):
                        for eid_to_mark in (source_id, target_id):
                            try:
                                await self._graph.update_entity(
                                    eid_to_mark,
                                    {"identity_core": 1},
                                    group_id=group_id,
                                )
                            except Exception:
                                pass  # May fail if entity already marked

            # Publish graph changes
            serialized_nodes = []
            serialized_edges = []
            for ent_data in result.entities:
                eid = entity_map.get(ent_data["name"])
                if eid:
                    ent_obj = await self._graph.get_entity(eid, group_id)
                    if ent_obj:
                        serialized_nodes.append(
                            {
                                "id": ent_obj.id,
                                "name": ent_obj.name,
                                "entityType": ent_obj.entity_type,
                                "summary": ent_obj.summary,
                                "activationCurrent": 0.0,
                                "accessCount": 0,
                                "lastAccessed": None,
                                "createdAt": (
                                    ent_obj.created_at.isoformat()
                                    if ent_obj.created_at
                                    else None
                                ),
                                "updatedAt": (
                                    ent_obj.updated_at.isoformat()
                                    if ent_obj.updated_at
                                    else None
                                ),
                            }
                        )
            for rel_data in result.relationships:
                src_name = rel_data.get("source") or rel_data.get("source_entity", "")
                tgt_name = rel_data.get("target") or rel_data.get("target_entity", "")
                src_id = entity_map.get(src_name)
                tgt_id = entity_map.get(tgt_name)
                if src_id and tgt_id:
                    rels = await self._graph.get_relationships(
                        src_id, direction="outgoing", group_id=group_id
                    )
                    for r in rels:
                        if r.target_id == tgt_id:
                            serialized_edges.append(
                                {
                                    "id": r.id,
                                    "source": r.source_id,
                                    "target": r.target_id,
                                    "predicate": r.predicate,
                                    "weight": r.weight,
                                    "validFrom": (
                                        r.valid_from.isoformat() if r.valid_from else None
                                    ),
                                    "validTo": (
                                        r.valid_to.isoformat() if r.valid_to else None
                                    ),
                                    "createdAt": (
                                        r.created_at.isoformat() if r.created_at else None
                                    ),
                                }
                            )
                            break
            self._publish(
                group_id,
                "graph.nodes_added",
                {
                    "episode_id": episode_id,
                    "entity_count": len(result.entities),
                    "relationship_count": len(result.relationships),
                    "new_entities": new_entity_names,
                    "nodes": serialized_nodes,
                    "edges": serialized_edges,
                },
            )

            # Embedding
            await self._update_episode_status(
                episode_id, EpisodeStatus.EMBEDDING, group_id=group_id,
            )
            try:
                for ent_data in result.entities:
                    eid = entity_map.get(ent_data["name"])
                    if eid:
                        ent = await self._graph.get_entity(eid, group_id)
                        if ent:
                            if self._cfg.structure_aware_embeddings:
                                await self._index_entity_with_structure(ent, group_id)
                            else:
                                await self._search.index_entity(ent)
                ep = await self._graph.get_episode_by_id(episode_id, group_id)
                if ep:
                    await self._search.index_episode(ep)
            except Exception as embed_err:
                logger.warning(
                    "Embedding failed for episode %s (non-fatal): %s",
                    episode_id, embed_err,
                )

            # Activating
            await self._update_episode_status(
                episode_id, EpisodeStatus.ACTIVATING, group_id=group_id,
            )

            # Complete
            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            await self._update_episode_status(
                episode_id, EpisodeStatus.COMPLETED, group_id=group_id,
                processing_duration_ms=elapsed_ms,
            )
            self._publish(
                group_id,
                "episode.completed",
                {
                    "episodeId": episode_id,
                    "status": "completed",
                    "entity_count": len(result.entities),
                    "relationship_count": len(result.relationships),
                    "duration_ms": elapsed_ms,
                },
            )
            logger.info(
                "Ingested episode %s: %d entities, %d relationships",
                episode_id, len(result.entities), len(result.relationships),
            )
            self.invalidate_briefing_cache(group_id)

        except Exception as e:
            logger.error("Failed to process episode %s: %s", episode_id, e)
            await self._update_episode_status(
                episode_id, EpisodeStatus.FAILED, group_id=group_id, error=str(e),
            )
            self._publish(
                group_id,
                "episode.failed",
                {"episodeId": episode_id, "status": "failed", "error": str(e)},
            )
            raise

    async def ingest_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Ingest a text episode: store, extract, resolve, link.

        Returns the episode ID. Thin wrapper over store_episode + project_episode.
        """
        episode_id = await self.store_episode(content, group_id, source, session_id)
        try:
            await self.project_episode(episode_id, group_id)
        except Exception:
            pass  # project_episode already sets FAILED status
        return episode_id

    async def _index_entity_with_structure(
        self,
        entity: Entity,
        group_id: str,
    ) -> None:
        """Build structure-aware embedding text that includes relationship predicates."""
        rels = await self._graph.get_relationships(
            entity.id,
            active_only=True,
            group_id=group_id,
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
            pred_natural = natural_names.get(r.predicate, r.predicate.lower().replace("_", " "))
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
                            sr.node_id,
                            "episode",
                            sr.score,
                            query,
                            now,
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
                    await self._publish_access_event(
                        sr.node_id, entity.name, entity.entity_type, group_id, "recall"
                    )

                    # Populate working memory buffer for entities
                    if self._working_memory is not None:
                        self._working_memory.add(
                            sr.node_id,
                            "entity",
                            sr.score,
                            query,
                            now,
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
                                "hop_distance": sr.hop_distance,
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
            search_hits = await self._search.search(query=name, group_id=group_id, limit=limit * 2)
            for eid, _score in search_hits:
                ent = await self._graph.get_entity(eid, group_id)
                if ent and (not entity_type or ent.entity_type == entity_type):
                    entities.append(ent)
                if len(entities) >= limit:
                    break

            # Fallback: if search returned nothing, try direct name lookup
            if not entities:
                entities = await self._graph.find_entities(
                    name=name, entity_type=entity_type, group_id=group_id, limit=limit
                )
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

            result.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "activation_score": round(activation_score, 4),
                    "created_at": entity.created_at.isoformat() if entity.created_at else None,
                    "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                    "access_count": access_count,
                }
            )

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
            search_hits = await self._search.search(query=query, group_id=group_id, limit=limit)
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
            result.append(
                {
                    "subject": source_name,
                    "predicate": r.predicate,
                    "object": target_name,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    "confidence": r.confidence,
                    "source_episode": r.source_episode,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )

        return result

    # ─── Forget entity ──────────────────────────────────────────────

    async def forget_entity(
        self,
        entity_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Soft-delete an entity and clear its activation."""
        entities = await self._graph.find_entities(name=entity_name, group_id=group_id, limit=1)
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
            subject_id,
            direction="outgoing",
            predicate=predicate,
            active_only=True,
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
                "message": (f"No active fact found: {subject_name} —{predicate}→ {object_name}."),
            }

        await self._graph.invalidate_relationship(
            target_rel.id, datetime.utcnow(), group_id=group_id
        )

        logger.info(
            "Forgot fact %s —%s→ %s, reason: %s",
            subject_name,
            predicate,
            object_name,
            reason,
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

    async def _entity_to_context_data(
        self, entity_id: str, name: str, entity_type: str,
        summary: str, group_id: str, now: float,
        detail_level: str = "full",
    ) -> dict:
        """Build context data dict for a single entity with activation and facts.

        detail_level controls rendering resolution:
        - "full": name + type + activation + summary + attributes + up to 5 facts
        - "summary": name + type + summary + up to 2 facts
        - "mention": name + type only
        """
        result: dict = {
            "name": name,
            "type": entity_type,
            "detail_level": detail_level,
            "id": entity_id,
        }

        if detail_level == "mention":
            result["activation"] = 0.0
            result["summary"] = None
            result["facts"] = []
            result["attributes"] = None
            return result

        from engram.activation.engine import compute_activation

        state = await self._activation.get_activation(entity_id)
        act = 0.0
        if state:
            act = compute_activation(state.access_history, now, self._cfg)
        result["activation"] = act
        result["summary"] = summary

        max_facts = 5 if detail_level == "full" else 2
        facts: list[str] = []
        rels = await self._graph.get_relationships(
            entity_id, active_only=True, group_id=group_id,
        )
        for r in rels[:max_facts]:
            src = await self.resolve_entity_name(r.source_id, group_id)
            tgt = await self.resolve_entity_name(r.target_id, group_id)
            facts.append(f"{src} {r.predicate} {tgt}")
        result["facts"] = facts

        # Only fetch attributes for full detail
        if detail_level == "full":
            entity = await self._graph.get_entity(entity_id, group_id)
            result["attributes"] = entity.attributes if entity else None
        else:
            result["attributes"] = None

        return result

    @staticmethod
    def _render_tier(header: str, entities: list[dict], facts: list[str]) -> str:
        """Render a single context tier as markdown with variable resolution.

        Each entity dict may have a 'detail_level' key:
        - "full": name + type + activation + summary + attributes + facts
        - "summary": name + type + summary + facts (no attributes)
        - "mention": name + type only
        """
        lines = [header, ""]
        for ed in entities:
            detail = ed.get("detail_level", "full")

            if detail == "mention":
                lines.append(f"- {ed['name']} ({ed['type']})")
                continue

            summary_part = f" — {ed['summary']}" if ed.get("summary") else ""
            # Append top attributes inline (full detail only)
            if detail == "full":
                attrs = ed.get("attributes")
                if attrs:
                    attr_parts = [f"{k}: {v}" for k, v in list(attrs.items())[:5]]
                    summary_part += f" [{', '.join(attr_parts)}]"
            lines.append(
                f"- {ed['name']} ({ed['type']}, act={ed['activation']:.2f})"
                f"{summary_part}"
            )
            # Render per-entity facts inline
            for fact in ed.get("facts", []):
                lines.append(f"  - {fact}")

        # Also render any tier-level facts not already covered by entities
        entity_facts = set()
        for ed in entities:
            entity_facts.update(ed.get("facts", []))
        extra_facts = [f for f in facts if f not in entity_facts]
        if extra_facts:
            for fact in extra_facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4

    def invalidate_briefing_cache(self, group_id: str) -> None:
        """Clear briefing cache entries for the given group."""
        keys_to_remove = [k for k in self._briefing_cache if k[0] == group_id]
        for k in keys_to_remove:
            del self._briefing_cache[k]

    async def _synthesize_briefing(
        self, structured_context: str, group_id: str, topic_hint: str | None,
    ) -> str:
        """Call Claude Haiku to synthesize a brief narrative from structured context."""
        import asyncio

        cache_key = (group_id, topic_hint)
        now = time.time()
        if cache_key in self._briefing_cache:
            ts, text = self._briefing_cache[cache_key]
            if now - ts < self._cfg.briefing_cache_ttl_seconds:
                return text

        try:
            import anthropic

            client = anthropic.Anthropic()

            def _call() -> str:
                resp = client.messages.create(
                    model=self._cfg.briefing_model,
                    max_tokens=self._cfg.briefing_max_tokens,
                    system=(
                        "You synthesize memory context into a brief, natural-sounding "
                        "summary for an AI assistant about to start a conversation. "
                        "Write 2-3 sentences: who the user is, what they're working on, "
                        "and relevant recent context. Be concise and conversational. "
                        "Do not mention activation scores or system internals."
                    ),
                    messages=[{"role": "user", "content": structured_context}],
                )
                return resp.content[0].text

            briefing = await asyncio.to_thread(_call)
            self._briefing_cache[cache_key] = (now, briefing)
            return briefing
        except Exception:
            logger.warning("Briefing synthesis failed, falling back to structured", exc_info=True)
            return structured_context

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        """Build a tiered markdown context summary of the most activated memories.

        Tiers:
        1. Identity Core — always-included identity entities + top relationships
        2. Project Context — topic-biased entities (from project_path or topic_hint)
        3. Recent Activity — top-activated entities filling remaining budget
        """
        from pathlib import Path

        from engram.activation.engine import compute_activation

        now = time.time()
        seen_ids: set[str] = set()

        # Derive topic_hint from project_path if not provided
        if project_path and not topic_hint:
            p = Path(project_path).expanduser()
            if p.name and str(p) != str(Path.home()):
                topic_hint = p.name

        # ── Layer 1: Identity Core ──
        layer1_entities: list[dict] = []
        layer1_facts: list[str] = []

        if self._cfg.identity_core_enabled and hasattr(self._graph, "get_identity_core_entities"):
            try:
                core_entities = await self._graph.get_identity_core_entities(group_id)
                for ce in core_entities:
                    ed = await self._entity_to_context_data(
                        ce.id, ce.name, ce.entity_type,
                        ce.summary or "", group_id, now,
                        detail_level="full",
                    )
                    layer1_entities.append(ed)
                    layer1_facts.extend(ed["facts"])
                    seen_ids.add(ce.id)
            except Exception:
                logger.debug("Identity core lookup failed (non-fatal)", exc_info=True)

        layer1_entities.sort(key=lambda x: x["activation"], reverse=True)
        layer1_text = self._render_tier("## Identity", layer1_entities, layer1_facts)

        # ── Layer 2: Project Context ──
        layer2_entities: list[dict] = []
        layer2_facts: list[str] = []

        if topic_hint:
            results = await self.recall(query=topic_hint, group_id=group_id, limit=15)
            for r in results:
                if r.get("result_type") == "episode":
                    continue
                ent = r["entity"]
                if ent["id"] in seen_ids:
                    continue
                # Variable resolution based on hop distance
                hop = r.get("score_breakdown", {}).get("hop_distance")
                if hop is None or hop == 0:
                    detail = "full"
                elif hop == 1:
                    detail = "summary"
                else:
                    detail = "mention"
                ed = await self._entity_to_context_data(
                    ent["id"], ent["name"], ent["type"],
                    ent.get("summary") or "", group_id, now,
                    detail_level=detail,
                )
                layer2_entities.append(ed)
                layer2_facts.extend(ed["facts"])
                seen_ids.add(ent["id"])
            layer2_entities.sort(key=lambda x: x["activation"], reverse=True)

        if layer2_entities:
            layer2_text = self._render_tier(
                f"## Project Context ({topic_hint})", layer2_entities, layer2_facts,
            )
        else:
            layer2_text = ""

        # ── Layer 3: Recent Activity ──
        layer3_entities: list[dict] = []
        layer3_facts: list[str] = []

        top = await self._activation.get_top_activated(group_id=group_id, limit=20)
        for eid, state in top:
            if eid in seen_ids:
                continue
            entity = await self._graph.get_entity(eid, group_id)
            if not entity:
                continue
            act = compute_activation(state.access_history, now, self._cfg)
            ed = await self._entity_to_context_data(
                entity.id, entity.name, entity.entity_type,
                entity.summary or "", group_id, now,
                detail_level="summary",
            )
            ed["activation"] = act  # use fresh computation
            layer3_entities.append(ed)
            layer3_facts.extend(ed["facts"])
            seen_ids.add(eid)

        layer3_entities.sort(key=lambda x: x["activation"], reverse=True)
        layer3_text = self._render_tier("## Recent Activity", layer3_entities, layer3_facts)

        # ── Assemble ──
        all_entities = layer1_entities + layer2_entities + layer3_entities
        all_facts = layer1_facts + layer2_facts + layer3_facts
        # Deduplicate facts
        seen_facts: set[str] = set()
        unique_facts: list[str] = []
        for f in all_facts:
            if f not in seen_facts:
                seen_facts.add(f)
                unique_facts.append(f)

        sections = [s for s in [layer1_text, layer2_text, layer3_text] if s]
        context_text = (
            "\n\n".join(sections)
            if sections
            else "## Active Memory Context\n\nNo memories loaded."
        )

        # Token estimate and truncation
        token_estimate = self._estimate_tokens(context_text)
        if token_estimate > max_tokens:
            char_budget = max_tokens * 4
            context_text = context_text[:char_budget]
            token_estimate = max_tokens

        # Record access for included entities
        for ed in all_entities:
            await self._activation.record_access(ed["id"], now, group_id=group_id)
            await self._publish_access_event(
                ed["id"], ed["name"], ed["type"], group_id, "context",
            )

        # Briefing format
        if format == "briefing" and self._cfg.briefing_enabled and all_entities:
            briefing = await self._synthesize_briefing(context_text, group_id, topic_hint)
            return {
                "context": briefing,
                "entity_count": len(all_entities),
                "fact_count": len(unique_facts),
                "token_estimate": self._estimate_tokens(briefing),
                "format": "briefing",
            }

        return {
            "context": context_text,
            "entity_count": len(all_entities),
            "fact_count": len(unique_facts),
            "token_estimate": token_estimate,
            "format": "structured",
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
                top_activated.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "summary": entity.summary,
                        "activation": round(act, 4),
                        "access_count": state.access_count,
                    }
                )

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
                        edges.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "predicate": r.predicate,
                                "weight": r.weight,
                            }
                        )
            result["edges"] = edges

        return result
