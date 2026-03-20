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

        try:
            pending = await graph_store.get_pending_evidence(
                group_id=group_id,
                limit=200,
            )
            pending = [
                ev
                for ev in pending
                if not ev.get("adjudication_request_id")
                and ev.get("commit_reason") != "needs_adjudication"
            ]
            if not pending:
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

                # Bare proper_name entities require cross-episode corroboration
                ev_signals = ev.get("corroborating_signals") or []
                if (
                    not forced
                    and ev.get("fact_class") == "entity"
                    and "proper_name" in ev_signals
                    and "identity_pattern" not in ev_signals
                ):
                    ev_key = self._evidence_key(ev)
                    group_count = len(groups.get(ev_key, []))
                    if group_count < 2:
                        if not dry_run:
                            await self._defer_evidence(
                                graph_store, ev, group_id=group_id
                            )
                            records.append(
                                EvidenceAdjudicationRecord(
                                    cycle_id=cycle_id,
                                    group_id=group_id,
                                    evidence_id=ev["evidence_id"],
                                    action="deferred",
                                    new_confidence=ev["confidence"],
                                    reason="proper_name_needs_corroboration",
                                ),
                            )
                        continue

                if forced or meets_threshold:
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
    ) -> None:
        new_cycles = ev.get("deferred_cycles", 0) + 1
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
