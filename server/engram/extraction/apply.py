"""Shared graph apply path for projection bundles."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime

from engram.config import ActivationConfig
from engram.entity_dedup_policy import (
    IDENTIFIER_ENTITY_TYPE,
    dedup_policy,
    normalize_extracted_entity_type,
    should_promote_entity_type_to_identifier,
)
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.conflicts import (
    get_contradictory_predicates,
    is_exclusive_predicate,
)
from engram.extraction.models import ApplyOutcome, ClaimCandidate, EntityCandidate
from engram.extraction.resolver import resolve_entity_fast
from engram.extraction.temporal import resolve_temporal_hint
from engram.models.consolidation import RelationshipApplyResult
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.protocols import ActivationStore, GraphStore
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


def merge_entity_attributes(
    existing: Entity,
    new_summary: str | None,
    new_pii: bool = False,
    new_pii_categories: list[str] | None = None,
    new_attributes: dict | None = None,
) -> dict:
    """Merge new attributes into an existing entity. Returns update dict."""
    updates: dict = {}

    from engram.graph_manager import GraphManager

    if new_summary and new_summary != existing.summary:
        if GraphManager._is_meta_summary(new_summary):
            entity_type = getattr(existing, "entity_type", "Other") or "Other"
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

    if new_attributes:
        merged_attrs = {**(existing.attributes or {}), **new_attributes}
        if merged_attrs != (existing.attributes or {}):
            updates["attributes"] = json.dumps(merged_attrs)

    if new_pii and not existing.pii_detected:
        updates["pii_detected"] = 1

    if new_pii_categories:
        existing_cats = existing.pii_categories or []
        merged_cats = list(set(existing_cats + new_pii_categories))
        if merged_cats != existing_cats:
            updates["pii_categories"] = json.dumps(merged_cats)

    return updates


def resolve_relationship_temporals(
    rel_data: dict,
    dt_now: datetime,
) -> tuple[datetime, datetime | None, float]:
    """Resolve temporal fields for an extracted relationship."""
    valid_from_str = rel_data.get("valid_from")
    valid_to_str = rel_data.get("valid_to")
    temporal_hint = rel_data.get("temporal_hint")

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
            else:
                confidence = 0.7
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

    return valid_from, valid_to, confidence


async def apply_relationship_fact(
    graph_store,
    canonicalizer: PredicateCanonicalizer,
    cfg: ActivationConfig,
    rel_data: dict,
    entity_map: dict[str, str],
    group_id: str,
    source_episode: str,
) -> RelationshipApplyResult:
    """Apply extracted relationship semantics through one shared path."""
    source_name = (
        rel_data.get("source") or rel_data.get("source_entity") or rel_data.get("source_name") or ""
    )
    target_name = (
        rel_data.get("target") or rel_data.get("target_entity") or rel_data.get("target_name") or ""
    )
    source_id = rel_data.get("source_id") or entity_map.get(source_name)
    target_id = rel_data.get("target_id") or entity_map.get(target_name)

    if not source_id or not target_id:
        return RelationshipApplyResult(
            action="missing_entities",
            metadata={"source_name": source_name, "target_name": target_name},
        )

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
    predicate = canonicalizer.canonicalize(predicate)

    dt_now = utc_now()
    valid_from, valid_to, confidence = resolve_relationship_temporals(rel_data, dt_now)
    explicit_confidence = rel_data.get("confidence")
    if explicit_confidence is not None:
        try:
            confidence = min(1.0, max(0.0, float(explicit_confidence)))
        except (TypeError, ValueError):
            pass

    polarity = rel_data.get("polarity", "positive")
    if polarity not in ("positive", "negative", "uncertain"):
        polarity = "positive"

    rel_weight = float(rel_data.get("weight", 1.0))
    constraints_hit: list[str] = []

    if polarity == "negative":
        constraints_hit.append("negative_polarity")
        existing_rels = await graph_store.get_relationships(
            source_id,
            direction="outgoing",
            predicate=predicate,
            active_only=True,
            group_id=group_id,
        )
        for existing_rel in existing_rels:
            if existing_rel.target_id == target_id:
                await graph_store.invalidate_relationship(
                    existing_rel.id,
                    dt_now,
                    group_id=group_id,
                )
                logger.info(
                    "Negation invalidated relationship %s (%s -> %s via %s)",
                    existing_rel.id,
                    existing_rel.source_id,
                    existing_rel.target_id,
                    existing_rel.predicate,
                )
    elif polarity == "uncertain":
        rel_weight *= 0.5

    if is_exclusive_predicate(predicate):
        constraints_hit.append("exclusive_predicate")
        conflicts = await graph_store.find_conflicting_relationships(
            source_id,
            predicate,
            group_id,
        )
        for conflict in conflicts:
            if conflict.target_id == target_id:
                continue
            await graph_store.invalidate_relationship(
                conflict.id,
                valid_from,
                group_id=group_id,
            )
            logger.info(
                "Invalidated conflicting relationship %s (%s -> %s via %s)",
                conflict.id,
                conflict.source_id,
                conflict.target_id,
                conflict.predicate,
            )

    if polarity == "positive":
        contra_preds = get_contradictory_predicates(predicate)
        for contra in contra_preds:
            constraints_hit.append("contradictory_predicate")
            contra_rels = await graph_store.get_relationships(
                source_id,
                direction="outgoing",
                predicate=contra,
                active_only=True,
                group_id=group_id,
            )
            for contra_rel in contra_rels:
                if contra_rel.target_id == target_id:
                    await graph_store.invalidate_relationship(
                        contra_rel.id,
                        dt_now,
                        group_id=group_id,
                    )
                    logger.info(
                        "Contradictory invalidated %s (%s -> %s via %s) due to new %s edge",
                        contra_rel.id,
                        contra_rel.source_id,
                        contra_rel.target_id,
                        contra_rel.predicate,
                        predicate,
                    )

        existing_rel = await graph_store.find_existing_relationship(
            source_id,
            target_id,
            predicate,
            group_id,
        )
        if existing_rel:
            if rel_weight > existing_rel.weight:
                await graph_store.update_relationship_weight(
                    source_id,
                    target_id,
                    rel_weight - existing_rel.weight,
                    group_id=group_id,
                    predicate=predicate,
                )
                action = "updated_existing"
            else:
                action = "duplicate_skipped"
            logger.debug(
                "Skipped duplicate relationship %s -> %s via %s",
                source_id,
                target_id,
                predicate,
            )
            return RelationshipApplyResult(
                source_id=source_id,
                target_id=target_id,
                predicate=predicate,
                polarity=polarity,
                confidence=confidence,
                weight=rel_weight,
                action=action,
                created=False,
                constraints_hit=constraints_hit + ["existing_duplicate"],
                metadata={"existing_relationship_id": existing_rel.id},
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
        source_episode=source_episode,
        group_id=group_id,
    )
    await graph_store.create_relationship(rel)

    if cfg.identity_core_enabled and predicate in cfg.identity_predicates:
        for eid_to_mark in (source_id, target_id):
            try:
                await graph_store.update_entity(
                    eid_to_mark,
                    {"identity_core": 1},
                    group_id=group_id,
                )
            except Exception:
                logger.warning("Failed to mark identity core for %s", eid_to_mark, exc_info=True)

    return RelationshipApplyResult(
        source_id=source_id,
        target_id=target_id,
        predicate=predicate,
        polarity=polarity,
        confidence=confidence,
        weight=rel_weight,
        action="created",
        created=True,
        constraints_hit=constraints_hit,
        metadata={"relationship_id": rel.id},
    )


class ApplyEngine:
    """Resolve typed candidates into graph writes."""

    def __init__(
        self,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cfg: ActivationConfig,
        canonicalizer: PredicateCanonicalizer,
        *,
        publish_access_event: Callable[[str, str, str, str, str], Awaitable[None]] | None = None,
        conv_context=None,
        labile_tracker=None,
        event_publisher: Callable[[str, str, dict | None], None] | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._canonicalizer = canonicalizer
        self._publish_access_event = publish_access_event
        self._conv_context = conv_context
        self._labile_tracker = labile_tracker
        self._event_publisher = event_publisher

    async def apply_entities(
        self,
        candidates: list[EntityCandidate],
        episode: Episode,
        group_id: str,
        *,
        recall_content: str | None = None,
    ) -> ApplyOutcome:
        """Resolve and write entity candidates for an episode."""
        outcome = ApplyOutcome()
        session_entities: dict[str, Entity] = {}
        now = time.time()
        content = recall_content or episode.content

        async def _get_candidates(name: str, gid: str) -> list[Entity]:
            return await self._graph.find_entity_candidates(name, gid)

        for candidate in candidates:
            name = candidate.name
            entity_type, _identifier_form = normalize_extracted_entity_type(
                name,
                candidate.entity_type,
            )
            summary = candidate.summary
            attributes = candidate.attributes or None
            pii_detected = candidate.pii_detected
            pii_categories = candidate.pii_categories

            if candidate.epistemic_mode == "meta":
                logger.debug("Skipping meta-tagged entity %r (type=%s)", name, entity_type)
                outcome.meta_entity_names.add(name)
                continue

            existing_entity = await resolve_entity_fast(
                name,
                entity_type,
                _get_candidates,
                group_id,
                session_entities=session_entities,
            )

            if existing_entity:
                entity_id = existing_entity.id
                updates = merge_entity_attributes(
                    existing_entity,
                    summary,
                    pii_detected,
                    pii_categories,
                    new_attributes=attributes,
                )
                identifier_decision = dedup_policy(name, existing_entity.name)
                if (
                    identifier_decision.exact_identifier_match
                    and entity_type == IDENTIFIER_ENTITY_TYPE
                    and existing_entity.entity_type != IDENTIFIER_ENTITY_TYPE
                    and should_promote_entity_type_to_identifier(existing_entity.entity_type)
                ):
                    updates["entity_type"] = IDENTIFIER_ENTITY_TYPE
                # Track provenance — record this episode as evidence
                existing_sources = existing_entity.source_episode_ids or []
                if episode.id not in existing_sources:
                    updated_sources = existing_sources + [episode.id]
                    updates["source_episode_ids"] = json.dumps(updated_sources)
                    updates["evidence_count"] = len(updated_sources)
                    ep_ts = episode.created_at
                    span_start = existing_entity.evidence_span_start
                    if span_start is None or ep_ts < span_start:
                        updates["evidence_span_start"] = ep_ts.isoformat()
                    span_end = existing_entity.evidence_span_end
                    if span_end is None or ep_ts > span_end:
                        updates["evidence_span_end"] = ep_ts.isoformat()

                if updates:
                    await self._graph.update_entity(entity_id, updates, group_id=group_id)

                if self._labile_tracker is not None:
                    from engram.retrieval.reconsolidation import attempt_reconsolidation

                    labile = self._labile_tracker.get_labile(existing_entity.id)
                    if labile and not self._labile_tracker.is_budget_exceeded(
                        existing_entity.id,
                        self._cfg.reconsolidation_max_modifications,
                    ):
                        recon_updates = attempt_reconsolidation(
                            existing_entity,
                            content,
                            labile,
                            self._cfg,
                        )
                        if recon_updates is not None:
                            await self._graph.update_entity(
                                existing_entity.id,
                                recon_updates,
                                group_id,
                            )
                            self._labile_tracker.record_modification(existing_entity.id)
                            await self._activation.record_access(
                                existing_entity.id,
                                now,
                                group_id=group_id,
                            )
                            recon_attrs = (
                                existing_entity.attributes
                                if isinstance(existing_entity.attributes, dict)
                                else {}
                            )
                            recon_attrs["recon_count"] = recon_attrs.get("recon_count", 0) + 1
                            recon_attrs["recon_last"] = now
                            await self._graph.update_entity(
                                existing_entity.id,
                                {"attributes": json.dumps(recon_attrs)},
                                group_id,
                            )
                            if self._event_publisher is not None:
                                self._event_publisher(
                                    group_id,
                                    "entity.reconsolidated",
                                    {
                                        "entity_id": existing_entity.id,
                                        "entity_name": existing_entity.name,
                                    },
                                )
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
                    source_episode_ids=[episode.id],
                    evidence_count=1,
                    evidence_span_start=episode.created_at,
                    evidence_span_end=episode.created_at,
                )
                await self._graph.create_entity(entity)
                session_entities[entity_id] = entity
                outcome.new_entity_names.append(name)

            outcome.entity_map[name] = entity_id
            await self._graph.link_episode_entity(episode.id, entity_id)
            await self._activation.record_access(entity_id, now, group_id=group_id)
            if self._publish_access_event is not None:
                await self._publish_access_event(
                    entity_id,
                    name,
                    entity_type,
                    group_id,
                    "ingest",
                )

            if self._conv_context is not None and self._cfg.conv_session_entity_seeds_enabled:
                self._conv_context.add_session_entity(
                    entity_id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    weight_increment=1.0,
                    now=now,
                )

        return outcome

    async def apply_relationships(
        self,
        claims: list[ClaimCandidate],
        *,
        entity_map: dict[str, str],
        meta_entity_names: set[str],
        group_id: str,
        source_episode: str,
    ) -> list[RelationshipApplyResult]:
        """Apply typed claim candidates through shared relationship semantics."""
        results: list[RelationshipApplyResult] = []
        for claim in claims:
            if (
                claim.subject_text in meta_entity_names
                or (claim.object_text or "") in meta_entity_names
            ):
                logger.debug(
                    "Skipping relationship %s->%s (meta entity involved)",
                    claim.subject_text,
                    claim.object_text,
                )
                continue

            rel_result = await apply_relationship_fact(
                graph_store=self._graph,
                canonicalizer=self._canonicalizer,
                cfg=self._cfg,
                rel_data=claim.raw_payload
                or {
                    "source": claim.subject_text,
                    "target": claim.object_text,
                    "predicate": claim.predicate,
                    "polarity": claim.polarity,
                    "temporal_hint": claim.temporal_hint,
                    "confidence": claim.confidence,
                },
                entity_map=entity_map,
                group_id=group_id,
                source_episode=source_episode,
            )
            results.append(rel_result)
        return results
