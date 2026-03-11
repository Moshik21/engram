"""Post-apply hooks for projection orchestration."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from engram.config import ActivationConfig
from engram.extraction.models import ApplyOutcome, ProjectionBundle
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship

logger = logging.getLogger(__name__)


class ProjectionPostProcessor:
    """Run non-extraction hooks after bundle apply."""

    def __init__(
        self,
        *,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        update_episode_status: Callable[[str, EpisodeStatus], Awaitable[None]],
        index_entity_with_structure: Callable[[Entity, str], Awaitable[None]],
        list_intentions: Callable[[str], Awaitable[list]],
        update_intention_fire: Callable[[str, str, str], Awaitable[None]],
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._update_episode_status = update_episode_status
        self._index_entity_with_structure = index_entity_with_structure
        self._list_intentions = list_intentions
        self._update_intention_fire = update_intention_fire

    async def apply_bootstrap_part_of_edges(
        self,
        episode: Episode,
        entity_map: dict[str, str],
        group_id: str,
    ) -> None:
        if episode.source != "auto:bootstrap" or not entity_map:
            return

        import re as _re

        match = _re.match(r"\[project-bootstrap\|([^|]+)\|", episode.content or "")
        if not match:
            return

        project_name = match.group(1)
        projects = await self._graph.find_entities(
            name=project_name,
            entity_type="Project",
            group_id=group_id,
            limit=1,
        )
        if not projects:
            return

        project_id = projects[0].id
        for entity_id in entity_map.values():
            if entity_id == project_id:
                continue
            rel = Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=entity_id,
                target_id=project_id,
                predicate="PART_OF",
                weight=0.8,
                source_episode=episode.id,
                group_id=group_id,
            )
            await self._graph.create_relationship(rel)

    async def run_surprise_detection(
        self,
        *,
        entity_map: dict[str, str],
        group_id: str,
        now: float,
        surprise_cache,
    ) -> None:
        if not self._cfg.surprise_detection_enabled or surprise_cache is None or not entity_map:
            return

        try:
            from engram.retrieval.surprise import detect_surprises

            surprises = await detect_surprises(
                entity_ids=list(entity_map.values()),
                graph_store=self._graph,
                activation_store=self._activation,
                cfg=self._cfg,
                group_id=group_id,
                now=now,
            )
            if surprises:
                surprise_cache.put(
                    group_id,
                    surprises[: self._cfg.surprise_max_per_episode],
                    now,
                )
        except Exception as surprise_err:
            logger.debug("Surprise detection failed (non-fatal): %s", surprise_err)

    async def run_prospective_memory(
        self,
        *,
        content: str,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> list:
        if not self._cfg.prospective_memory_enabled or not entity_map:
            return []

        try:
            if self._cfg.prospective_graph_embedded:
                intention_entities = await self._list_intentions(group_id)
                if intention_entities:
                    from engram.activation.spreading import spread_activation
                    from engram.retrieval.prospective import check_intention_activations

                    seeds = [(entity_id, 0.5) for entity_id in entity_map.values()]
                    spreading_bonuses, _ = await spread_activation(
                        seeds,
                        self._graph,
                        self._cfg,
                        group_id,
                    )
                    intention_ids = [entity.id for entity in intention_entities]
                    states = await self._activation.batch_get(intention_ids)
                    trigger_matches = await check_intention_activations(
                        spreading_results=spreading_bonuses,
                        activation_states=states,
                        intention_entities=intention_entities,
                        extracted_entity_ids=set(entity_map.values()),
                        now=time.time(),
                        cfg=self._cfg,
                        max_per_episode=self._cfg.prospective_max_per_episode,
                    )
                    if trigger_matches:
                        for match in trigger_matches:
                            await self._update_intention_fire(
                                match.intention_id,
                                group_id,
                                episode_id,
                            )
                        return trigger_matches
            else:
                intentions = await self._graph.list_intentions(group_id)
                if intentions:
                    from engram.retrieval.prospective import check_triggers

                    embed_fn = None
                    provider = getattr(self._search, "_provider", None)
                    if provider and hasattr(provider, "embed_query"):
                        embed_fn = provider.embed_query

                    trigger_matches = await check_triggers(
                        content=content,
                        entity_names=list(entity_map.keys()),
                        intentions=intentions,
                        embed_fn=embed_fn,
                    )
                    if trigger_matches:
                        for match in trigger_matches[: self._cfg.prospective_max_per_episode]:
                            await self._graph.increment_intention_fire_count(
                                match.intention_id,
                                group_id,
                            )
                        return trigger_matches[: self._cfg.prospective_max_per_episode]
        except Exception as prospective_err:
            logger.debug(
                "Prospective trigger check failed (non-fatal): %s",
                prospective_err,
            )
        return []

    async def publish_graph_changes(
        self,
        *,
        bundle: ProjectionBundle,
        apply_outcome: ApplyOutcome,
        group_id: str,
        episode_id: str,
        publish_event: Callable[[str, str, dict | None], None],
    ) -> None:
        serialized_nodes = []
        for candidate in bundle.entities:
            entity_id = apply_outcome.entity_map.get(candidate.name)
            if not entity_id:
                continue
            entity = await self._graph.get_entity(entity_id, group_id)
            if entity is None:
                continue
            serialized_nodes.append(
                {
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
            )

        serialized_edges = []
        seen_edge_ids: set[str] = set()
        for rel_result in apply_outcome.relationship_results:
            if not rel_result.source_id or not rel_result.target_id:
                continue
            rels = await self._graph.get_relationships(
                rel_result.source_id,
                direction="outgoing",
                predicate=rel_result.predicate,
                group_id=group_id,
            )
            for rel in rels:
                if rel.target_id != rel_result.target_id or rel.id in seen_edge_ids:
                    continue
                seen_edge_ids.add(rel.id)
                serialized_edges.append(
                    {
                        "id": rel.id,
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "predicate": rel.predicate,
                        "weight": rel.weight,
                        "validFrom": rel.valid_from.isoformat() if rel.valid_from else None,
                        "validTo": rel.valid_to.isoformat() if rel.valid_to else None,
                        "createdAt": rel.created_at.isoformat() if rel.created_at else None,
                    }
                )
                break

        publish_event(
            group_id,
            "graph.nodes_added",
            {
                "episode_id": episode_id,
                "entity_count": len(bundle.entities),
                "relationship_count": len(bundle.claims),
                "new_entities": apply_outcome.new_entity_names,
                "nodes": serialized_nodes,
                "edges": serialized_edges,
            },
        )

    async def index_projected_bundle(
        self,
        *,
        bundle: ProjectionBundle,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> None:
        await self._update_episode_status(
            episode_id,
            EpisodeStatus.EMBEDDING,
        )
        try:
            for candidate in bundle.entities:
                entity_id = entity_map.get(candidate.name)
                if not entity_id:
                    continue
                entity = await self._graph.get_entity(entity_id, group_id)
                if entity is None:
                    continue
                if self._cfg.structure_aware_embeddings:
                    await self._index_entity_with_structure(entity, group_id)
                else:
                    await self._search.index_entity(entity)

            episode = await self._graph.get_episode_by_id(episode_id, group_id)
            if episode:
                await self._search.index_episode(episode)
        except Exception as embed_err:
            logger.warning(
                "Embedding failed for episode %s (non-fatal): %s",
                episode_id,
                embed_err,
            )

    async def store_emotional_encoding_context(
        self,
        *,
        episode_id: str,
        content: str,
        entity_map: dict[str, str],
        group_id: str,
    ) -> None:
        if not self._cfg.emotional_salience_enabled:
            return

        from engram.extraction.salience import compute_emotional_salience

        salience = compute_emotional_salience(content)
        encoding_ctx = json.dumps(
            {
                "arousal": round(salience.arousal, 4),
                "self_reference": round(salience.self_reference, 4),
                "social_density": round(salience.social_density, 4),
                "narrative_tension": round(salience.narrative_tension, 4),
                "composite": round(salience.composite, 4),
            }
        )
        await self._graph.update_episode(
            episode_id,
            {"encoding_context": encoding_ctx},
            group_id=group_id,
        )

        for entity_id in entity_map.values():
            try:
                entity = await self._graph.get_entity(entity_id, group_id)
                if entity is None:
                    continue
                raw = entity.attributes
                attrs = dict(raw) if isinstance(raw, dict) else {}
                attrs["emo_arousal"] = round(salience.arousal, 4)
                attrs["emo_self_ref"] = round(salience.self_reference, 4)
                attrs["emo_social"] = round(salience.social_density, 4)
                attrs["emo_composite"] = round(salience.composite, 4)
                await self._graph.update_entity(
                    entity_id,
                    {"attributes": json.dumps(attrs)},
                    group_id=group_id,
                )
            except Exception:
                pass
