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
from engram.extraction.resolver import resolve_entity_fast, validate_entity_name
from engram.extraction.temporal import resolve_temporal_hint
from engram.models.consolidation import RelationshipApplyResult
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.protocols import ActivationStore, GraphStore
from engram.utils.dates import utc_now
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)


# Importance-as-a-prior seeds (cfg.importance_prior_enabled, EVAL-GATED, default
# off). Seeded into ActivationState.consolidated_strength at entity commit so
# one-shot high-value facts (identity, durable types, client remember()
# proposals) keep a floor inside the ACT-R ln-sum instead of decaying like
# chatter. Bounded: total consolidated_strength never exceeds the cap, so
# repeated commits of the same entity cannot inflate the prior.
_IMPORTANCE_SEED_IDENTITY = 0.02
_IMPORTANCE_SEED_DURABLE = 0.01
_IMPORTANCE_SEED_CLIENT_PROPOSAL = 0.01
_IMPORTANCE_SEED_CAP = 0.05
_IMPORTANCE_DURABLE_TYPES = frozenset(
    {"Decision", "Preference", "Commitment", "Correction", "Goal"}
)


# Supersession (cfg.supersession_enabled, M3.4, EVAL-GATED, default off).
# Additional exclusive-by-nature predicate classes, on top of the always-on
# EXCLUSIVE_PREDICATES set (which already covers the location/employment/role
# classes post-canonicalization: LIVES_IN -> LOCATED_IN, EMPLOYED_BY ->
# WORKS_AT, etc.). When the flag is on, a new edge in one of these classes
# sets valid_to on prior ACTIVE same-source same-predicate different-target
# edges instead of leaving both live ("moved to Denver" ends Seattle).
# Known scope risk the eval gate exists for: PREFERS and USES_VERSION are
# exclusive per (source, predicate) only when the source has a single slot —
# "prefers coffee" then "prefers Python" would cross-supersede.
_SUPERSESSION_EXCLUSIVE_PREDICATES = frozenset({"USES_VERSION", "NAMED", "IS_NAMED", "PREFERS"})


# Role/title attribute keys that all denote a person's (current) role or title.
# Extraction emits these inconsistently (role / new_role / job_title / position /
# title), which left stale and current values COEXISTING on the entity (e.g.
# Priya: role="engineer" + new_role="Director of Research"). Collapse them to a
# single canonical "role" key so a newer assertion overwrites the prior via
# merge_entity_attributes' {**existing, **new}. Update-intent prefixes
# (new_/current_/latest_) win an in-dict collision.
_ROLE_ATTR_KEYS = {
    "role",
    "new_role",
    "current_role",
    "latest_role",
    "title",
    "new_title",
    "current_title",
    "job_title",
    "job",
    "position",
    "new_position",
    "occupation",
}


def _canonicalize_attribute_keys(attrs: dict | None) -> dict | None:
    """Collapse inconsistent role/title attribute keys to a canonical 'role'."""
    if not attrs:
        return None
    out: dict = {}
    role_value = None
    role_priority = -1
    for key, value in attrs.items():
        if str(key).lower() in _ROLE_ATTR_KEYS:
            priority = 1 if str(key).lower().startswith(("new_", "current_", "latest_")) else 0
            if priority >= role_priority:
                role_value = value
                role_priority = priority
        else:
            out[key] = value
    if role_value is not None:
        out["role"] = role_value
    return out or None


def merge_entity_attributes(
    existing: Entity,
    new_summary: str | None,
    new_pii: bool = False,
    new_pii_categories: list[str] | None = None,
    new_attributes: dict | None = None,
) -> dict:
    """Merge new attributes into an existing entity. Returns update dict."""
    updates: dict = {}

    if new_summary and new_summary != existing.summary:
        if is_meta_summary(new_summary):
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
                # Keep the most RECENT fragments (the tail), not the oldest. The
                # summary is appended chronologically, so the current state lives
                # at the end; truncating the head (the prior behavior kept the
                # head and dropped the tail) would discard the current value and
                # retain stale text -- the opposite of "carry current values".
                merged = "..." + merged[-497:]
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


async def _auto_create_endpoint(
    graph_store,
    name: str,
    entity_map: dict[str, str],
    group_id: str,
    *,
    entity_type: str = "Concept",
    client_proposal: bool = False,
) -> str | None:
    """Materialize a minimal provisional endpoint entity for a dropped edge.

    Returns the new entity id and records it in entity_map so the relationship
    can persist, or None when the name is not a plausible entity (so the caller
    falls back to the existing missing_entities drop). Gated by the caller on
    cfg.graph_auto_create_endpoints.
    """
    from engram.extraction.narrow.entity_extractor import _STOPWORDS

    stripped = (name or "").strip()
    if (
        not validate_entity_name(
            stripped,
            entity_type=entity_type,
            client_proposal=client_proposal,
        )
        or len(stripped) < 3
    ):
        return None
    if stripped.lower() in {w.lower() for w in _STOPWORDS}:
        return None

    entity_id = f"ent_{uuid.uuid4().hex[:12]}"
    now = utc_now()
    # Prefer Decision-shaped types for long client-promoted statement names.
    resolved_type = entity_type
    if client_proposal and resolved_type == "Concept" and len(stripped.split()) >= 4:
        resolved_type = "Decision"
    entity = Entity(
        id=entity_id,
        name=stripped,
        entity_type=resolved_type,
        group_id=group_id,
        evidence_count=1,
        evidence_span_start=now,
        evidence_span_end=now,
        attributes={"provisional_endpoint": True},
    )
    await graph_store.create_entity(entity)
    entity_map[name] = entity_id
    return entity_id


async def apply_relationship_fact(
    graph_store,
    canonicalizer: PredicateCanonicalizer,
    cfg: ActivationConfig,
    rel_data: dict,
    entity_map: dict[str, str],
    group_id: str,
    source_episode: str,
    *,
    conversation_date: datetime | None = None,
) -> RelationshipApplyResult:
    """Apply extracted relationship semantics through one shared path."""
    # Client proposals carry {subject, predicate, object}; the legacy/extractor
    # path carries {source, target, predicate}. Accept both so a proposed edge
    # between two committed entities is never silently dropped as missing_entities.
    source_name = (
        rel_data.get("source")
        or rel_data.get("source_entity")
        or rel_data.get("source_name")
        or rel_data.get("subject")
        or ""
    )
    target_name = (
        rel_data.get("target")
        or rel_data.get("target_entity")
        or rel_data.get("target_name")
        or rel_data.get("object")
        or ""
    )
    source_id = rel_data.get("source_id") or entity_map.get(source_name)
    target_id = rel_data.get("target_id") or entity_map.get(target_name)

    auto_created_endpoints: list[str] = []
    # Client-proposal edges often reference Decision statement names longer than
    # the narrow-extractor 5-word limit; treat them as client proposals so
    # auto-create accepts statement-length endpoints.
    proposal_edge = bool(
        rel_data.get("signals")
        and (
            "client_proposal" in (rel_data.get("signals") or [])
            or "high_signal_type" in (rel_data.get("signals") or [])
        )
    ) or bool(rel_data.get("client_proposal"))
    if (not source_id or not target_id) and cfg.graph_auto_create_endpoints:
        if not source_id:
            source_id = await _auto_create_endpoint(
                graph_store,
                source_name,
                entity_map,
                group_id,
                client_proposal=proposal_edge,
            )
            if source_id:
                auto_created_endpoints.append(source_name)
        if not target_id:
            source_type_hint = (
                "Decision" if proposal_edge and len(target_name.split()) >= 4 else "Concept"
            )
            target_id = await _auto_create_endpoint(
                graph_store,
                target_name,
                entity_map,
                group_id,
                entity_type=source_type_hint,
                client_proposal=proposal_edge or len(target_name.split()) > 5,
            )
            if target_id:
                auto_created_endpoints.append(target_name)

    if not source_id or not target_id:
        return RelationshipApplyResult(
            action="missing_entities",
            metadata={"source_name": source_name, "target_name": target_name},
        )

    if source_id == target_id:
        # A relationship whose endpoints resolve to the same entity is a garbage
        # self-loop (e.g. "Guitar -FOCUSES_ON-> Guitar"). Drop it before persist.
        logger.debug(
            "Dropping self-loop relationship %s -> %s (%s)",
            source_name,
            target_name,
            source_id,
        )
        return RelationshipApplyResult(
            action="self_loop_dropped",
            created=False,
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
    # Anchor relative temporal hints to when the conversation happened, not to
    # ingest time. A "since last month" hint on a back-dated remember() must
    # resolve against the episode's conversation_date or it silently lands today.
    reference_date = conversation_date or dt_now
    valid_from, valid_to, confidence = resolve_relationship_temporals(rel_data, reference_date)
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

    superseded_edge_ids: list[str] = []
    if is_exclusive_predicate(predicate) or (
        cfg.supersession_enabled and predicate in _SUPERSESSION_EXCLUSIVE_PREDICATES
    ):
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
            if cfg.supersession_enabled:
                superseded_edge_ids.append(conflict.id)
            logger.info(
                "Invalidated conflicting relationship %s (%s -> %s via %s)",
                conflict.id,
                conflict.source_id,
                conflict.target_id,
                conflict.predicate,
            )
    # Provenance for a future undo/audit: the superseding apply result carries
    # which edges it closed and when. The Relationship model/stores have no
    # edge-attribute column, so provenance lives on the apply-result metadata
    # (which feeds evidence/audit records), not on the persisted edge.
    supersession_meta: dict = {}
    if superseded_edge_ids:
        constraints_hit.append("superseded_prior")
        supersession_meta = {
            "superseded_edge_ids": superseded_edge_ids,
            "superseded_at": valid_from.isoformat(),
        }

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
                metadata={"existing_relationship_id": existing_rel.id, **supersession_meta},
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
    persisted_id = await graph_store.create_relationship(rel)
    if not persisted_id:
        # The store could not persist the edge (e.g. an endpoint entity was not
        # resolvable in the backend). Report not-created so persisted counts stay
        # honest instead of counting a phantom edge.
        return RelationshipApplyResult(
            source_id=source_id,
            target_id=target_id,
            predicate=predicate,
            polarity=polarity,
            confidence=confidence,
            weight=rel_weight,
            action="persist_failed",
            created=False,
            constraints_hit=constraints_hit + ["persist_failed"],
            metadata={"relationship_id": rel.id, **supersession_meta},
        )

    if cfg.identity_core_enabled and predicate in cfg.identity_predicates:
        for eid_to_mark in (source_id, target_id):
            try:
                mark_updates: dict = {"identity_core": 1}
                # Identity-core auto-promotes to the semantic tier in the same write.
                marked = await graph_store.get_entity(eid_to_mark, group_id)
                if marked is not None:
                    mark_attrs = dict(marked.attributes or {})
                    if mark_attrs.get("mat_tier") != "semantic":
                        mark_attrs["mat_tier"] = "semantic"
                        mark_updates["attributes"] = json.dumps(mark_attrs)
                await graph_store.update_entity(
                    eid_to_mark,
                    mark_updates,
                    group_id=group_id,
                )
            except Exception:
                logger.warning("Failed to mark identity core for %s", eid_to_mark, exc_info=True)

    metadata: dict = {"relationship_id": rel.id, **supersession_meta}
    if auto_created_endpoints:
        metadata["auto_created_endpoints"] = auto_created_endpoints
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
        metadata=metadata,
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

    async def _seed_importance_prior(
        self,
        entity_id: str,
        entity_type: str,
        *,
        identity_core: bool,
        client_proposal: bool,
    ) -> None:
        """Seed a bounded importance prior on the entity's activation state.

        Routes through the activation store (get/set) so the graph write path
        stays decoupled from activation internals. Gated by the caller on
        cfg.importance_prior_enabled.
        """
        from engram.activation.engine import seed_consolidated_strength
        from engram.models.activation import ActivationState

        if identity_core:
            seed = _IMPORTANCE_SEED_IDENTITY
        elif entity_type in _IMPORTANCE_DURABLE_TYPES:
            seed = _IMPORTANCE_SEED_DURABLE
        elif client_proposal:
            seed = _IMPORTANCE_SEED_CLIENT_PROPOSAL
        else:
            return
        state = await self._activation.get_activation(entity_id)
        if state is None:
            state = ActivationState(node_id=entity_id)
        if seed_consolidated_strength(state, seed, _IMPORTANCE_SEED_CAP):
            await self._activation.set_activation(entity_id, state)

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
        # M1.6 (F10): an entity committed from user content is an environmental
        # mention (tier="mentioned", w_ranking=0.1). Bootstrap-artifact episodes
        # are documentation, not user mentions — they record at the surfaced
        # (hygiene-only) tier. mentioned_ids guards at most one mentioned event
        # per (entity, episode) within this call; re-commits of the same episode
        # are caught by the source_episode_ids idempotency check at the record
        # site (episode.id already present ⇒ not a first mention).
        is_bootstrap = episode.source == "auto:bootstrap"
        mentioned_ids: set[str] = set()

        async def _get_candidates(name: str, gid: str) -> list[Entity]:
            return await self._graph.find_entity_candidates(name, gid)

        for candidate in candidates:
            name = candidate.name
            # Normalize first so lowercase extractor output ("decision") gets
            # the same typed semantics as canonical TitleCase downstream.
            entity_type, _identifier_form = normalize_extracted_entity_type(
                name,
                candidate.entity_type,
            )
            signals = []
            if isinstance(candidate.raw_payload, dict):
                signals = list(candidate.raw_payload.get("signals") or [])
            is_client_proposal = (
                "client_proposal" in signals
                or "high_signal_type" in signals
                or entity_type
                in {
                    "Decision",
                    "Preference",
                    "Correction",
                    "Goal",
                    "Commitment",
                }
            )
            if not validate_entity_name(
                name,
                entity_type=entity_type,
                client_proposal=is_client_proposal,
            ):
                logger.debug("Skipping invalid entity name %r", name)
                continue
            summary = candidate.summary
            attributes = _canonicalize_attribute_keys(candidate.attributes)
            pii_detected = candidate.pii_detected
            pii_categories = candidate.pii_categories
            from engram.extraction.promotion import should_protect_as_identity_core

            protect_identity = self._cfg.identity_core_enabled and should_protect_as_identity_core(
                entity_type,
                client_proposal=is_client_proposal,
            )

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

                if protect_identity and not getattr(existing_entity, "identity_core", False):
                    updates["identity_core"] = 1
                    # Identity-core auto-promotes to the semantic tier in the
                    # same update-entity write.
                    promo_attrs = (
                        json.loads(updates["attributes"])
                        if "attributes" in updates
                        else dict(existing_entity.attributes or {})
                    )
                    if promo_attrs.get("mat_tier") != "semantic":
                        promo_attrs["mat_tier"] = "semantic"
                        updates["attributes"] = json.dumps(promo_attrs)
                # Trust path: do not silently overwrite/append identity_core summaries.
                # Compare EXISTING vs PROPOSED (candidate summary) — never the merged
                # "old; new" string from merge_entity_attributes (substring false-negative).
                if (
                    getattr(existing_entity, "identity_core", False)
                    and is_client_proposal
                    and summary
                ):
                    from engram.extraction.promotion import identity_core_summary_conflict

                    if identity_core_summary_conflict(
                        existing_entity.summary,
                        summary,
                        entity_type=entity_type,
                    ):
                        from engram.extraction.harness_metrics import (
                            record_client_proposal_outcomes,
                        )

                        record_client_proposal_outcomes(identity_conflicts=1)
                        updates.pop("summary", None)
                        logger.info(
                            "Blocked identity_core summary overwrite for %r "
                            "(proposed conflicted with protected summary)",
                            name,
                        )
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
                if protect_identity:
                    # Identity-core auto-promotes to the semantic tier at creation
                    # (attributes carry the tier so the lite backend round-trips it).
                    attributes = {**(attributes or {}), "mat_tier": "semantic"}
                entity = Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    summary=summary,
                    attributes=attributes,
                    group_id=group_id,
                    pii_detected=pii_detected,
                    pii_categories=pii_categories,
                    identity_core=protect_identity,
                    mat_tier="semantic" if protect_identity else "episodic",
                    source_episode_ids=[episode.id],
                    evidence_count=1,
                    evidence_span_start=episode.created_at,
                    evidence_span_end=episode.created_at,
                )
                await self._graph.create_entity(entity)
                session_entities[entity_id] = entity
                outcome.new_entity_names.append(name)

            outcome.entity_map[name] = entity_id
            await self._graph.link_episode_entity(episode.id, entity_id, group_id=group_id)
            already_evidenced = existing_entity is not None and episode.id in (
                existing_entity.source_episode_ids or []
            )
            if is_bootstrap or already_evidenced or entity_id in mentioned_ids:
                tier = "surfaced"
            else:
                tier = "mentioned"
                mentioned_ids.add(entity_id)
            await self._activation.record_access(entity_id, now, group_id=group_id, tier=tier)
            if self._cfg.importance_prior_enabled:
                await self._seed_importance_prior(
                    entity_id,
                    entity_type,
                    identity_core=protect_identity
                    or bool(existing_entity and getattr(existing_entity, "identity_core", False)),
                    client_proposal=is_client_proposal,
                )
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
        conversation_date: datetime | None = None,
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
                # Append a sentinel instead of `continue` so results stay
                # positionally aligned with `claims`. committed_id_map zips the
                # two together; dropping a result here shifts every later
                # evidence_id onto the wrong relationship_id.
                results.append(RelationshipApplyResult(action="skipped_meta"))
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
                conversation_date=conversation_date,
            )
            results.append(rel_result)
        return results
