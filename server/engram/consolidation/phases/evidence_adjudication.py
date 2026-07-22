"""Evidence adjudication phase — materializes corroborated evidence."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.extraction.commit_policy import AdaptiveCommitPolicy, CommitThresholds
from engram.graph_manager import EvidenceMaterializationFailure
from engram.models.consolidation import CycleContext, EvidenceAdjudicationRecord, PhaseResult

logger = logging.getLogger(__name__)


class EvidenceAdjudicationPhase(ConsolidationPhase):
    """Warm-tier phase that promotes unresolved evidence into graph state."""

    def __init__(self, graph_manager: Any | None = None) -> None:
        self._manager = graph_manager

    @property
    def name(self) -> str:
        return "evidence_adjudication"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.evidence_extraction_enabled or not cfg.evidence_store_deferred:
            return set()
        return {"get_pending_evidence", "update_evidence_status", "get_entity_count"}

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
    ) -> tuple[PhaseResult, list[EvidenceAdjudicationRecord]]:
        if not cfg.evidence_extraction_enabled:
            return PhaseResult(phase=self.name, status="skipped"), []
        if not cfg.evidence_store_deferred:
            return PhaseResult(phase=self.name, status="skipped"), []

        start = time.monotonic()
        records: list[EvidenceAdjudicationRecord] = []
        materialized = 0
        drain_rejected = 0
        collapsed_rejected = 0
        stale_rejected = 0

        # Autonomic deferred queue maintenance (no external LLM).
        # Without this, intake permanently outruns a fixed 200-row promote budget.
        try:
            from engram.consolidation.evidence_drain import (
                load_deferred_evidence,
                reject_evidence_rows,
                reject_junk_evidence,
                scaled_drain_budget,
                select_redundant_entity_evidence,
                select_stale_low_value_evidence,
            )

            base_budget = int(
                getattr(cfg, "consolidation_evidence_drain_max_per_cycle", 500) or 500
            )
            max_budget = int(getattr(cfg, "consolidation_evidence_drain_max_budget", 5000) or 5000)
            deferred_rows = await load_deferred_evidence(graph_store, group_id)
            drain_budget = scaled_drain_budget(
                len(deferred_rows),
                base_budget=base_budget,
                max_budget=max_budget,
            )
            if deferred_rows:
                # 1) Classified junk first (fill budget with rejectable rows).
                drain_result = await reject_junk_evidence(
                    graph_store,
                    group_id=group_id,
                    rows=deferred_rows,
                    dry_run=dry_run,
                    batch_size=min(200, drain_budget),
                    prioritize_junk=True,
                    max_reject=drain_budget,
                )
                drain_rejected = int(drain_result.get("rejected") or 0)
                # Re-load after junk drain when applying so collapses see remaining.
                if drain_rejected and not dry_run:
                    deferred_rows = await load_deferred_evidence(graph_store, group_id)
                elif dry_run and drain_rejected:
                    from engram.consolidation.evidence_drain import (
                        classify_deferred_evidence,
                    )

                    deferred_rows = [
                        r
                        for r in deferred_rows
                        if classify_deferred_evidence(r).disposition != "reject_junk"
                    ]

                # 2) Already-present entity names — deferred duplicates of graph facts.
                exists_budget = int(
                    getattr(
                        cfg,
                        "consolidation_evidence_already_exists_max_per_cycle",
                        500,
                    )
                    or 0
                )
                if exists_budget > 0 and deferred_rows:
                    existing_names = await self._existing_entity_names(
                        graph_store, group_id, deferred_rows
                    )
                    redundant = select_redundant_entity_evidence(
                        deferred_rows,
                        existing_names,
                        limit=exists_budget,
                    )
                    if redundant:
                        collapse = await reject_evidence_rows(
                            graph_store,
                            group_id=group_id,
                            rows=redundant,
                            reason_prefix="drain_already_exists",
                            dry_run=dry_run,
                            reason_for_row=lambda _r: "entity_name_present",
                        )
                        collapsed_rejected = int(collapse.get("rejected") or 0)
                        if collapsed_rejected and not dry_run:
                            deferred_rows = await load_deferred_evidence(graph_store, group_id)
                        elif dry_run and collapsed_rejected:
                            rid = {r.get("evidence_id") for r in redundant}
                            deferred_rows = [
                                r for r in deferred_rows if r.get("evidence_id") not in rid
                            ]

                # 3) Stale low-value deferred that never corroborated.
                stale_budget = int(
                    getattr(cfg, "consolidation_evidence_stale_max_per_cycle", 500) or 0
                )
                if stale_budget > 0 and deferred_rows:
                    stale = select_stale_low_value_evidence(
                        deferred_rows,
                        max_age_days=float(
                            getattr(cfg, "consolidation_evidence_stale_reject_days", 21.0) or 21.0
                        ),
                        min_deferred_cycles=int(
                            getattr(cfg, "evidence_forced_commit_cycles", 5) or 5
                        ),
                        limit=stale_budget,
                    )
                    if stale:
                        stale_result = await reject_evidence_rows(
                            graph_store,
                            group_id=group_id,
                            rows=stale,
                            reason_prefix="drain_stale",
                            dry_run=dry_run,
                            reason_for_row=lambda _r: "stale_uncorroborated",
                        )
                        stale_rejected = int(stale_result.get("rejected") or 0)

                total_drained = drain_rejected + collapsed_rejected + stale_rejected
                if total_drained:
                    logger.info(
                        "Evidence autonomic drain: junk=%d already_exists=%d "
                        "stale=%d deferred_pool=%d budget=%d dry_run=%s",
                        drain_rejected,
                        collapsed_rejected,
                        stale_rejected,
                        len(deferred_rows),
                        drain_budget,
                        dry_run,
                    )
                drain_rejected = total_drained
        except Exception:
            logger.debug("Evidence autonomic drain skipped", exc_info=True)

        try:
            adj_limit = int(getattr(cfg, "consolidation_evidence_adjudication_limit", 200) or 200)
            # Under debt, process more open rows so deferred_cycles actually advance
            # for the long tail (not only the top-confidence 200 forever).
            if drain_rejected > 0:
                adj_limit = min(2000, max(adj_limit, drain_rejected))
            pending = await graph_store.get_pending_evidence(
                group_id=group_id,
                limit=adj_limit,
            )
            pending = [
                ev
                for ev in pending
                if not ev.get("adjudication_request_id")
                and ev.get("commit_reason") != "needs_adjudication"
            ]
            if not pending:
                if drain_rejected > 0:
                    return PhaseResult(
                        phase=self.name,
                        status="success",
                        items_processed=drain_rejected,
                        items_affected=drain_rejected,
                        duration_ms=(time.monotonic() - start) * 1000,
                    ), []
                return PhaseResult(
                    phase=self.name,
                    status="skipped",
                    duration_ms=(time.monotonic() - start) * 1000,
                ), []

            if self._manager is None and not dry_run:
                return (
                    PhaseResult(
                        phase=self.name,
                        status="error",
                        error="graph_manager_required",
                        duration_ms=(time.monotonic() - start) * 1000,
                    ),
                    [],
                )
            manager = self._manager
            if not dry_run:
                assert manager is not None

            groups: dict[str, list[dict]] = defaultdict(list)
            for ev in pending:
                groups[self._evidence_key(ev)].append(ev)

            entity_count = await graph_store.get_entity_count(group_id)
            policy = AdaptiveCommitPolicy(
                thresholds=CommitThresholds(
                    entity=cfg.evidence_commit_entity_threshold,
                    relationship=cfg.evidence_commit_relationship_threshold,
                    attribute=cfg.evidence_commit_attribute_threshold,
                    temporal=cfg.evidence_commit_temporal_threshold,
                ),
            )

            for evidence_key, evidence_group in groups.items():
                if len(evidence_group) <= 1:
                    continue
                corroboration_boost = min(0.15, 0.05 * (len(evidence_group) - 1))
                for ev in evidence_group:
                    new_confidence = min(1.0, ev["confidence"] + corroboration_boost)
                    if new_confidence <= ev["confidence"]:
                        continue
                    if not dry_run:
                        await graph_store.update_evidence_status(
                            ev["evidence_id"],
                            ev["status"],
                            updates={"confidence": new_confidence},
                            group_id=group_id,
                        )
                    ev["confidence"] = new_confidence
                    if not dry_run:
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="corroborated",
                                new_confidence=new_confidence,
                                reason=f"corroborated_by_{len(evidence_group)}_sources",
                            ),
                        )

            promotable_by_episode: dict[str, list[dict]] = defaultdict(list)
            for ev in pending:
                if ev["status"] == "approved":
                    promotable_by_episode[ev["episode_id"]].append(ev)
                    continue

                cycles = ev.get("deferred_cycles", 0)
                forced = cycles >= cfg.evidence_forced_commit_cycles
                threshold = policy._effective_threshold(
                    ev["fact_class"],
                    entity_count,
                    signals=ev.get("corroborating_signals"),
                )
                meets_threshold = ev["confidence"] >= threshold

                # Bare proper_name entities require cross-episode corroboration.
                # Unverified single-source client proposals are held to the same bar:
                # an annotation whose source_span never verified must corroborate
                # across episodes before it commits, so it cannot weaponize a high
                # caller-tier confidence into a first-sight commit.
                ev_signals = ev.get("corroborating_signals") or []
                needs_corroboration_reason = None
                if "proper_name" in ev_signals and "identity_pattern" not in ev_signals:
                    needs_corroboration_reason = "proper_name_needs_corroboration"
                elif "observation_sourced" in ev_signals and "identity_pattern" not in ev_signals:
                    # M1.4 squatter guard: observation-sourced entities need
                    # >=2-episode corroboration for full commit.
                    needs_corroboration_reason = "observation_needs_corroboration"
                elif (
                    ev.get("source_type") == "client_proposal" and "span_verified" not in ev_signals
                ):
                    needs_corroboration_reason = "unverified_proposal_needs_corroboration"

                if (
                    not forced
                    and ev.get("fact_class") == "entity"
                    and needs_corroboration_reason is not None
                ):
                    ev_key = self._evidence_key(ev)
                    group_count = len(groups.get(ev_key, []))
                    if group_count < 2:
                        if not dry_run:
                            # Waiting-for-corroboration is not a failed
                            # adjudication: the proper-name hold must not
                            # consume the cycles>=5 forced-reject window (I2).
                            await self._defer_evidence(
                                graph_store,
                                ev,
                                group_id=group_id,
                                count_cycle=needs_corroboration_reason
                                not in (
                                    "proper_name_needs_corroboration",
                                    "observation_needs_corroboration",
                                ),
                            )
                            records.append(
                                EvidenceAdjudicationRecord(
                                    cycle_id=cycle_id,
                                    group_id=group_id,
                                    evidence_id=ev["evidence_id"],
                                    action="deferred",
                                    new_confidence=ev["confidence"],
                                    reason=needs_corroboration_reason,
                                ),
                            )
                        continue

                if forced or meets_threshold:
                    from engram.consolidation.evidence_drain import (
                        should_force_commit_evidence,
                    )

                    force_high_signal_only = bool(
                        getattr(
                            cfg,
                            "consolidation_evidence_force_commit_high_signal_only",
                            True,
                        )
                    )
                    # Aged pattern sludge must die, not force-materialize into the graph.
                    if (
                        forced
                        and force_high_signal_only
                        and not meets_threshold
                        and not should_force_commit_evidence(ev)
                    ):
                        if not dry_run:
                            await graph_store.update_evidence_status(
                                ev["evidence_id"],
                                "rejected",
                                updates={
                                    "commit_reason": (
                                        f"drain_stale:forced_reject_after_{cycles}_cycles"
                                    ),
                                },
                                group_id=group_id,
                            )
                            records.append(
                                EvidenceAdjudicationRecord(
                                    cycle_id=cycle_id,
                                    group_id=group_id,
                                    evidence_id=ev["evidence_id"],
                                    action="rejected",
                                    new_confidence=ev["confidence"],
                                    reason=f"forced_reject_after_{cycles}_cycles",
                                ),
                            )
                        continue

                    reason = (
                        f"forced_after_{cycles}_cycles" if forced else "promoted_by_adjudication"
                    )
                    if not dry_run:
                        await graph_store.update_evidence_status(
                            ev["evidence_id"],
                            "approved",
                            updates={"commit_reason": reason},
                            group_id=group_id,
                        )
                    ev["status"] = "approved"
                    ev["commit_reason"] = reason
                    promotable_by_episode[ev["episode_id"]].append(ev)
                    if not dry_run:
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="approved",
                                new_confidence=ev["confidence"],
                                reason=reason,
                            ),
                        )
                    continue

                if not dry_run:
                    await self._defer_evidence(
                        graph_store,
                        ev,
                        group_id=group_id,
                    )
                    records.append(
                        EvidenceAdjudicationRecord(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            evidence_id=ev["evidence_id"],
                            action="deferred",
                            new_confidence=ev["confidence"],
                            reason="below_threshold",
                        ),
                    )

            if dry_run:
                elapsed = (time.monotonic() - start) * 1000
                return (
                    PhaseResult(
                        phase=self.name,
                        status="success",
                        items_processed=len(pending),
                        items_affected=0,
                        duration_ms=elapsed,
                    ),
                    [],
                )

            assert manager is not None
            for episode_id, evidence_rows in promotable_by_episode.items():
                try:
                    outcome = await manager.materialize_stored_evidence(
                        episode_id,
                        evidence_rows,
                        group_id=group_id,
                    )
                except EvidenceMaterializationFailure as exc:
                    for ev in evidence_rows:
                        await self._defer_evidence(
                            graph_store,
                            ev,
                            group_id=group_id,
                        )
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="materialization_failed",
                                new_confidence=ev["confidence"],
                                reason=str(exc),
                            ),
                        )
                    continue

                if not outcome.materialized:
                    for ev in evidence_rows:
                        await self._defer_evidence(
                            graph_store,
                            ev,
                            group_id=group_id,
                        )
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="deferred",
                                new_confidence=ev["confidence"],
                                reason="not_materialized",
                            ),
                        )
                    continue

                for ev in evidence_rows:
                    committed_id = outcome.committed_ids.get(ev["evidence_id"])
                    if committed_id:
                        await graph_store.update_evidence_status(
                            ev["evidence_id"],
                            "committed",
                            updates={
                                "commit_reason": ev.get("commit_reason"),
                                "committed_id": committed_id,
                            },
                            group_id=group_id,
                        )
                        ev["status"] = "committed"
                        ev["committed_id"] = committed_id
                        materialized += 1
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="materialized",
                                new_confidence=ev["confidence"],
                                reason=ev.get("commit_reason") or "materialized",
                            ),
                        )
                    else:
                        await self._defer_evidence(
                            graph_store,
                            ev,
                            group_id=group_id,
                        )
                        records.append(
                            EvidenceAdjudicationRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                evidence_id=ev["evidence_id"],
                                action="deferred",
                                new_confidence=ev["confidence"],
                                reason="not_materialized",
                            ),
                        )

            elapsed = (time.monotonic() - start) * 1000
            return (
                PhaseResult(
                    phase=self.name,
                    status="success",
                    items_processed=len(pending),
                    items_affected=materialized,
                    duration_ms=elapsed,
                ),
                records,
            )

        except Exception as exc:
            logger.error("evidence_adjudication failed: %s", exc, exc_info=True)
            elapsed = (time.monotonic() - start) * 1000
            return (
                PhaseResult(
                    phase=self.name,
                    status="error",
                    error=str(exc),
                    duration_ms=elapsed,
                ),
                records,
            )

    async def _defer_evidence(
        self,
        graph_store: Any,
        ev: dict,
        *,
        group_id: str,
        count_cycle: bool = True,
    ) -> None:
        new_cycles = ev.get("deferred_cycles", 0) + (1 if count_cycle else 0)
        await graph_store.update_evidence_status(
            ev["evidence_id"],
            "deferred",
            updates={
                "deferred_cycles": new_cycles,
                "commit_reason": None,
                "committed_id": None,
            },
            group_id=group_id,
        )
        ev["status"] = "deferred"
        ev["deferred_cycles"] = new_cycles
        ev["commit_reason"] = None
        ev["committed_id"] = None

    async def _existing_entity_names(
        self,
        graph_store: Any,
        group_id: str,
        deferred_rows: list[dict],
    ) -> set[str]:
        """Resolve which deferred entity names already exist (best-effort)."""
        from engram.consolidation.evidence_drain import _entity_name

        names = {
            _entity_name(row)
            for row in deferred_rows
            if str(row.get("fact_class") or "") == "entity"
        }
        names.discard("")
        if not names:
            return set()

        existing: set[str] = set()
        find_candidates = getattr(graph_store, "find_entity_candidates", None)
        if not callable(find_candidates):
            return existing

        # Cap lookups so a huge deferred pile does not dominate the warm cycle.
        for name in list(names)[:500]:
            candidates = None
            try:
                candidates = await find_candidates(
                    name,
                    group_id=group_id,
                    limit=5,
                )
            except TypeError:
                try:
                    candidates = await find_candidates(name, group_id)
                except Exception:
                    candidates = None
            except Exception:
                candidates = None
            for cand in candidates or []:
                cand_name = getattr(cand, "name", None)
                if cand_name is None and isinstance(cand, dict):
                    cand_name = cand.get("name")
                if cand_name and str(cand_name).casefold() == name.casefold():
                    existing.add(name)
                    break
        return existing

    def _evidence_key(self, ev: dict) -> str:
        """Create a dedup key for corroboration."""
        payload = ev.get("payload", {})
        fc = ev.get("fact_class", "")
        if fc == "entity":
            return (
                f"entity:{payload.get('name', '').lower()}:{payload.get('entity_type', '').lower()}"
            )
        if fc == "relationship":
            return (
                f"rel:{payload.get('subject', '').lower()}"
                f":{payload.get('predicate', '')}"
                f":{payload.get('object', '').lower()}"
            )
        if fc == "attribute":
            return (
                f"attr:{payload.get('entity', '').lower()}"
                f":{payload.get('attribute_type', '')}"
                f":{str(payload.get('value', '')).lower()}"
            )
        return f"other:{ev.get('evidence_id', '')}"
