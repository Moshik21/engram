"""Episode replay phase: re-extract entities from recent episodes."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.resolver import resolve_entity
from engram.extraction.temporal import resolve_temporal_hint
from engram.models.consolidation import CycleContext, PhaseResult, ReplayRecord
from engram.models.entity import Entity
from engram.models.relationship import Relationship

logger = logging.getLogger(__name__)


class EpisodeReplayPhase(ConsolidationPhase):
    """Re-extract entities from recent COMPLETED episodes to find missed info."""

    def __init__(
        self,
        extractor: Any | None = None,
        canonicalizer: PredicateCanonicalizer | None = None,
    ) -> None:
        self._extractor = extractor
        self._canonicalizer = canonicalizer or PredicateCanonicalizer()

    @property
    def name(self) -> str:
        return "replay"

    async def execute(
        self,
        group_id: str,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[ReplayRecord]]:
        t0 = time.perf_counter()

        if not cfg.consolidation_replay_enabled:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=_elapsed_ms(t0),
            ), []

        # Skip replay if no graph changes occurred (tiered scheduling only)
        # Manual/pressure/scheduled triggers always run all phases fully
        if context is not None and context.trigger.startswith("tiered"):
            has_changes = bool(
                context.merge_survivor_ids
                or context.inferred_edge_entity_ids
                or context.replay_new_entity_ids
            )
            if not has_changes:
                logger.info("Replay: skipping — no graph changes from merge/infer")
                return PhaseResult(
                    phase=self.name,
                    status="skipped",
                    items_processed=0,
                    items_affected=0,
                    duration_ms=_elapsed_ms(t0),
                ), []

        if self._extractor is None:
            logger.warning("Replay phase: no EntityExtractor provided, skipping")
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=_elapsed_ms(t0),
            ), []

        max_per_cycle = cfg.consolidation_replay_max_per_cycle
        window_hours = cfg.consolidation_replay_window_hours
        min_age_hours = cfg.consolidation_replay_min_age_hours

        # Fetch recent episodes and filter in Python
        episodes = await graph_store.get_episodes(
            group_id=group_id,
            limit=max_per_cycle * 2,
        )

        now = datetime.utcnow()
        window_cutoff = now - timedelta(hours=window_hours)
        age_cutoff = now - timedelta(hours=min_age_hours)

        eligible = [
            ep
            for ep in episodes
            if (
                ep.status.value == "completed"
                and ep.created_at is not None
                and ep.created_at >= window_cutoff
                and ep.created_at <= age_cutoff
            )
        ][:max_per_cycle]

        if not eligible:
            return PhaseResult(
                phase=self.name,
                items_processed=0,
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        # If we have affected entity IDs from merge/infer, prioritize episodes linked to them
        if context and context.affected_entity_ids:
            affected_episodes = []
            for ep in eligible:
                try:
                    linked = set(await graph_store.get_episode_entities(ep.id))
                    if linked & context.affected_entity_ids:
                        affected_episodes.append(ep)
                except Exception:
                    pass
            if affected_episodes:
                eligible = affected_episodes[:max_per_cycle]
            # If no episodes overlap with affected entities, that's fine — use all eligible

        # Load existing entities once for resolution across all replays
        existing_entities = await graph_store.find_entities(
            group_id=group_id,
            limit=100000,
        )

        records: list[ReplayRecord] = []
        total_new_entities = 0
        total_new_rels = 0
        total_updated = 0

        for episode in eligible:
            try:
                record = await self._replay_episode(
                    episode=episode,
                    group_id=group_id,
                    graph_store=graph_store,
                    activation_store=activation_store,
                    search_index=search_index,
                    cfg=cfg,
                    cycle_id=cycle_id,
                    dry_run=dry_run,
                    context=context,
                    existing_entities=existing_entities,
                )
                records.append(record)
                total_new_entities += record.new_entities_found
                total_new_rels += record.new_relationships_found
                total_updated += record.entities_updated
            except Exception:
                logger.warning(
                    "Replay failed for episode %s (non-fatal)",
                    episode.id,
                    exc_info=True,
                )
                records.append(
                    ReplayRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        episode_id=episode.id,
                        skipped_reason="extraction_failed",
                    )
                )

        return PhaseResult(
            phase=self.name,
            items_processed=len(eligible),
            items_affected=total_new_entities + total_new_rels + total_updated,
            duration_ms=_elapsed_ms(t0),
        ), records

    async def _replay_episode(
        self,
        episode: Any,
        group_id: str,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
        existing_entities: list[Entity],
    ) -> ReplayRecord:
        """Re-extract a single episode and add genuinely new info."""
        # Skip re-extraction if episode's linked entities have no new neighbors
        # (no graph changes affect this episode's context)
        if context and context.affected_entity_ids:
            linked = set(await graph_store.get_episode_entities(episode.id))
            if linked and not (linked & context.affected_entity_ids):
                return ReplayRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    episode_id=episode.id,
                    skipped_reason="no_context_change",
                )

        result = await self._extractor.extract(episode.content)

        if not result.entities and not result.relationships:
            return ReplayRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                episode_id=episode.id,
                skipped_reason="no_new_info",
            )

        # Get entities already linked to this episode
        already_linked = set(await graph_store.get_episode_entities(episode.id))

        entity_map: dict[str, str] = {}
        new_entities = 0
        entities_updated = 0
        now_ts = time.time()

        for ent_data in result.entities:
            name = ent_data["name"]
            entity_type = ent_data.get("entity_type", "Other")
            summary = ent_data.get("summary")

            existing_entity = await resolve_entity(
                name,
                entity_type,
                existing_entities,
            )

            if existing_entity:
                entity_id = existing_entity.id
                entity_map[name] = entity_id

                if entity_id in already_linked:
                    continue

                # Existing entity not yet linked — merge attributes and link
                if not dry_run:
                    if summary:
                        from engram.graph_manager import GraphManager

                        updates = GraphManager._merge_entity_attributes(
                            existing_entity,
                            summary,
                            ent_data.get("pii_detected", False),
                            ent_data.get("pii_categories"),
                        )
                        if updates:
                            await graph_store.update_entity(
                                entity_id,
                                updates,
                                group_id=group_id,
                            )
                            entities_updated += 1

                    await graph_store.link_episode_entity(
                        episode.id,
                        entity_id,
                    )
                    await activation_store.record_access(
                        entity_id,
                        now_ts,
                        group_id=group_id,
                    )
                    if context is not None:
                        context.affected_entity_ids.add(entity_id)
            else:
                # Genuinely new entity
                entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                entity_map[name] = entity_id
                new_entities += 1

                if not dry_run:
                    entity = Entity(
                        id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        summary=summary,
                        group_id=group_id,
                        pii_detected=ent_data.get("pii_detected", False),
                        pii_categories=ent_data.get("pii_categories"),
                    )
                    await graph_store.create_entity(entity)
                    existing_entities.append(entity)
                    await graph_store.link_episode_entity(
                        episode.id,
                        entity_id,
                    )
                    await activation_store.record_access(
                        entity_id,
                        now_ts,
                        group_id=group_id,
                    )
                    await search_index.index_entity(entity)

                    if context is not None:
                        context.replay_new_entity_ids.add(entity_id)
                        context.affected_entity_ids.add(entity_id)

        # Process relationships
        new_rels = await self._replay_relationships(
            result.relationships,
            entity_map=entity_map,
            group_id=group_id,
            graph_store=graph_store,
            episode_id=episode.id,
            cycle_id=cycle_id,
            dry_run=dry_run,
        )

        return ReplayRecord(
            cycle_id=cycle_id,
            group_id=group_id,
            episode_id=episode.id,
            new_entities_found=new_entities,
            new_relationships_found=new_rels,
            entities_updated=entities_updated,
        )

    async def _replay_relationships(
        self,
        rel_data_list: list[dict],
        entity_map: dict[str, str],
        group_id: str,
        graph_store: Any,
        episode_id: str,
        cycle_id: str,
        dry_run: bool,
    ) -> int:
        """Create genuinely new relationships. Returns count of new rels."""
        new_count = 0

        for rel_data in rel_data_list:
            source_name = rel_data.get("source") or rel_data.get("source_entity", "")
            target_name = rel_data.get("target") or rel_data.get("target_entity", "")
            source_id = entity_map.get(source_name)
            target_id = entity_map.get(target_name)

            if not source_id or not target_id:
                continue

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

            # Check if this exact relationship already exists
            existing_rels = await graph_store.get_relationships(
                source_id,
                direction="outgoing",
                predicate=predicate,
                active_only=True,
                group_id=group_id,
            )
            already_exists = any(r.target_id == target_id for r in existing_rels)
            if already_exists:
                continue

            new_count += 1

            if dry_run:
                continue

            dt_now = datetime.utcnow()
            valid_from = dt_now
            valid_to = None

            valid_from_str = rel_data.get("valid_from")
            if valid_from_str:
                try:
                    valid_from = datetime.fromisoformat(valid_from_str)
                except (ValueError, TypeError):
                    resolved = resolve_temporal_hint(valid_from_str, dt_now)
                    if resolved:
                        valid_from = resolved

            valid_to_str = rel_data.get("valid_to")
            if valid_to_str:
                try:
                    valid_to = datetime.fromisoformat(valid_to_str)
                except (ValueError, TypeError):
                    resolved = resolve_temporal_hint(valid_to_str, dt_now)
                    if resolved:
                        valid_to = resolved

            rel = Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=source_id,
                target_id=target_id,
                predicate=predicate,
                weight=float(rel_data.get("weight", 1.0)),
                valid_from=valid_from,
                valid_to=valid_to,
                confidence=0.9,
                source_episode=f"replay:{cycle_id}:{episode_id}",
                group_id=group_id,
            )
            await graph_store.create_relationship(rel)

        return new_count


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
