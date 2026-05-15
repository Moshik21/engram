"""Structure-aware entity indexing for graph write paths."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.storage.protocols import GraphStore, SearchIndex


class StructureAwareEntityIndexer:
    """Build enriched search-index text from an entity and its graph context."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
    ) -> None:
        self._graph = graph_store
        self._search = search_index
        self._cfg = cfg

    async def index_entity(
        self,
        entity: Entity,
        group_id: str,
    ) -> None:
        """Index an entity using predicate-weighted relationship context."""
        rels = await self._graph.get_relationships(
            entity.id,
            active_only=True,
            group_id=group_id,
        )

        predicate_weights = self._cfg.predicate_weights
        default_weight = self._cfg.predicate_weight_default
        rels_sorted = sorted(
            rels,
            key=lambda rel: predicate_weights.get(rel.predicate, default_weight),
            reverse=True,
        )

        natural_names = self._cfg.predicate_natural_names
        max_rels = self._cfg.structure_max_relationships
        rel_parts: list[str] = []
        for rel in rels_sorted[:max_rels]:
            pred_natural = natural_names.get(
                rel.predicate,
                rel.predicate.lower().replace("_", " "),
            )
            if rel.source_id == entity.id:
                target = await self._graph.get_entity(rel.target_id, group_id)
                target_name = target.name if target else rel.target_id
                rel_parts.append(f"{entity.name} {pred_natural} {target_name}")
            else:
                source = await self._graph.get_entity(rel.source_id, group_id)
                source_name = source.name if source else rel.source_id
                rel_parts.append(f"{source_name} {pred_natural} {entity.name}")

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
