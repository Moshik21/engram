"""Extract entities from unextracted episodes and link known entities by name."""

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
from engram.models.consolidation import (
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    PhaseResult,
    ReplayRecord,
)
from engram.models.entity import Entity
from engram.models.episode import EpisodeProjectionState
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


class EpisodeReplayPhase(ConsolidationPhase):
    """Extract entities from unextracted episodes and link known entities by name."""

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

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.consolidation_replay_enabled:
            return set()
        return {
            "get_episodes",
            "get_episode_entities",
            "find_entities",
            "create_entity",
            "update_entity",
            "update_episode",
            "link_episode_entity",
            "create_relationship",
            "get_relationships",
            "invalidate_relationship",
            "find_conflicting_relationships",
            "find_existing_relationship",
        }

    def required_activation_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.consolidation_replay_enabled:
            return set()
        return {"record_access"}

    def required_search_index_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.consolidation_replay_enabled:
            return set()
        return {"index_entity"}

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

        now = utc_now()
        window_cutoff = now - timedelta(hours=window_hours)
        age_cutoff = now - timedelta(hours=min_age_hours)
        eligible = await self._load_eligible_episodes(
            graph_store=graph_store,
            group_id=group_id,
            max_per_cycle=max_per_cycle,
            window_cutoff=window_cutoff,
            age_cutoff=age_cutoff,
        )

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

        # Phase B: Graph-vocabulary linking — scan episodes for exact entity name matches
        if cfg.consolidation_replay_vocab_linking_enabled:
            vocab_records = await self._graph_vocabulary_link(
                episodes=eligible,
                group_id=group_id,
                graph_store=graph_store,
                existing_entities=existing_entities,
                cycle_id=cycle_id,
                dry_run=dry_run,
                context=context,
            )
            records.extend(vocab_records)

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

        extractor = self._extractor
        assert extractor is not None
        result = await extractor.extract(episode.content)

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
            cfg=cfg,
            episode_id=episode.id,
            cycle_id=cycle_id,
            dry_run=dry_run,
            context=context,
        )

        # Mark episode as projected after successful extraction
        if not dry_run and (new_entities > 0 or new_rels > 0 or entities_updated > 0):
            await graph_store.update_episode(
                episode.id,
                {
                    "projection_state": EpisodeProjectionState.PROJECTED.value,
                    "last_projection_reason": "replay_deferred_extraction",
                },
                group_id=group_id,
            )

        return ReplayRecord(
            cycle_id=cycle_id,
            group_id=group_id,
            episode_id=episode.id,
            new_entities_found=new_entities,
            new_relationships_found=new_rels,
            entities_updated=entities_updated,
        )

    async def _load_eligible_episodes(
        self,
        graph_store: Any,
        group_id: str,
        max_per_cycle: int,
        window_cutoff: datetime,
        age_cutoff: datetime,
    ) -> list[Any]:
        """Scan recent episodes until enough replay-eligible items are found."""
        eligible: list[Any] = []
        offset = 0
        batch_size = max(max_per_cycle * 2, 10)
        seen_episode_ids: set[str] = set()

        while len(eligible) < max_per_cycle:
            episodes = await graph_store.get_episodes(
                group_id=group_id,
                limit=batch_size,
                offset=offset,
            )
            if not episodes:
                break
            batch_ids = {ep.id for ep in episodes}
            if batch_ids and batch_ids.issubset(seen_episode_ids):
                break
            seen_episode_ids.update(batch_ids)
            offset += len(episodes)

            reached_window_end = False
            for ep in episodes:
                created_at = getattr(ep, "created_at", None)
                status = ep.status.value if hasattr(ep.status, "value") else ep.status

                if created_at is None:
                    continue
                if created_at < window_cutoff:
                    reached_window_end = True
                    break
                if status != "completed":
                    continue
                # Skip already-extracted episodes — narrow re-extraction is
                # deterministic waste.  Target CUE_ONLY / QUEUED episodes only.
                proj_state = getattr(ep, "projection_state", None)
                if proj_state is not None:
                    pval = proj_state.value if hasattr(proj_state, "value") else str(proj_state)
                    if pval == "projected":
                        continue
                if created_at > age_cutoff:
                    continue

                eligible.append(ep)
                if len(eligible) >= max_per_cycle:
                    break

            if reached_window_end:
                break

        return eligible[:max_per_cycle]

    async def _replay_relationships(
        self,
        rel_data_list: list[dict],
        entity_map: dict[str, str],
        group_id: str,
        graph_store: Any,
        cfg: ActivationConfig,
        episode_id: str,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
    ) -> int:
        """Apply replayed relationships through the ingestion relationship path."""
        from engram.graph_manager import GraphManager

        new_count = 0

        for rel_data in rel_data_list:
            if dry_run:
                source_name = rel_data.get("source") or rel_data.get("source_entity", "")
                target_name = rel_data.get("target") or rel_data.get("target_entity", "")
                if entity_map.get(source_name) and entity_map.get(target_name):
                    new_count += 1
                continue

            created = await GraphManager._apply_relationship_fact(
                graph_store=graph_store,
                canonicalizer=self._canonicalizer,
                cfg=cfg,
                rel_data=rel_data,
                entity_map=entity_map,
                group_id=group_id,
                source_episode=f"replay:{cycle_id}:{episode_id}",
            )
            if context is not None:
                trace = DecisionTrace(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    phase=self.name,
                    candidate_type="relationship",
                    candidate_id=_replay_candidate_id(
                        created.source_id,
                        created.target_id,
                        created.predicate,
                    ),
                    decision=created.action,
                    decision_source="shared_apply_path",
                    confidence=created.confidence,
                    threshold_band="applied" if created.created else "skipped",
                    features={
                        "polarity": created.polarity,
                        "weight": created.weight,
                        **created.metadata,
                    },
                    constraints_hit=created.constraints_hit,
                    metadata={"episode_id": episode_id},
                )
                context.add_decision_trace(trace)
                if created.created:
                    context.add_decision_outcome_label(
                        DecisionOutcomeLabel(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            phase=self.name,
                            decision_trace_id=trace.id,
                            outcome_type="materialization",
                            label="applied",
                            value=1.0,
                            metadata={"episode_id": episode_id},
                        )
                    )
            if created.created:
                new_count += 1

        return new_count


    async def _graph_vocabulary_link(
        self,
        episodes: list[Any],
        group_id: str,
        graph_store: Any,
        existing_entities: list[Entity],
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
    ) -> list[ReplayRecord]:
        """Scan episodes for exact substring matches of known entity names."""
        if not existing_entities or not episodes:
            return []

        # Build lookup sorted by name length descending (longer names match first)
        name_to_id: list[tuple[str, str]] = sorted(
            [(ent.name.lower(), ent.id) for ent in existing_entities if ent.name],
            key=lambda t: len(t[0]),
            reverse=True,
        )

        records: list[ReplayRecord] = []
        for ep in episodes:
            content = getattr(ep, "content", None)
            if not content:
                continue
            content_lower = content.lower()

            try:
                already_linked = set(await graph_store.get_episode_entities(ep.id))
            except Exception:
                already_linked = set()

            linked_count = 0
            for name_lower, entity_id in name_to_id:
                if entity_id in already_linked:
                    continue
                if name_lower in content_lower:
                    if not dry_run:
                        await graph_store.link_episode_entity(ep.id, entity_id)
                    already_linked.add(entity_id)
                    linked_count += 1
                    if context is not None:
                        context.affected_entity_ids.add(entity_id)

            if linked_count > 0:
                records.append(
                    ReplayRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        episode_id=ep.id,
                        new_entities_found=0,
                        new_relationships_found=0,
                        entities_updated=linked_count,
                        skipped_reason=None,
                    )
                )

        return records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _replay_candidate_id(
    source_id: str | None,
    target_id: str | None,
    predicate: str | None,
) -> str:
    if not source_id or not target_id or not predicate:
        return "missing"
    return f"{source_id}:{target_id}:{predicate}"
