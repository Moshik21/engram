"""Budgeted offline adjudication for unresolved ambiguous evidence."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult
from engram.utils.dates import utc_now, utc_now_iso

logger = logging.getLogger(__name__)


class EdgeAdjudicationPhase(ConsolidationPhase):
    """Offline phase that expires stale v3 edge-adjudication requests.

    Ambiguous evidence is resolved by the resident agent through the
    client-assisted adjudication tools; this phase only expires requests
    nobody answered within the TTL. The server-side LLM adjudicator was
    deleted 2026-09-04.
    """

    def __init__(self, graph_manager: Any | None = None) -> None:
        self._manager = graph_manager

    @property
    def name(self) -> str:
        return "edge_adjudication"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.edge_adjudication_enabled:
            return set()
        return {
            "get_pending_adjudication_requests",
            "get_episode_evidence",
            "update_adjudication_request",
            "update_evidence_status",
        }

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
    ) -> tuple[PhaseResult, list]:
        if not cfg.edge_adjudication_enabled or not cfg.evidence_extraction_enabled:
            return PhaseResult(phase=self.name, status="skipped"), []

        start = time.monotonic()
        requests = await graph_store.get_pending_adjudication_requests(
            group_id=group_id,
            limit=200,
        )
        if not requests:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=(time.monotonic() - start) * 1000,
            ), []

        processed = 0
        affected = 0
        now = utc_now()
        if dry_run:
            return (
                PhaseResult(
                    phase=self.name,
                    status="success",
                    items_processed=len(requests),
                    items_affected=0,
                    duration_ms=(time.monotonic() - start) * 1000,
                ),
                [],
            )

        for request in requests:
            processed += 1
            created_at = self._parse_dt(request.get("created_at")) or now
            age = now - created_at

            if age >= timedelta(hours=cfg.edge_adjudication_request_ttl_hours):
                affected += await self._expire_request(
                    graph_store,
                    request,
                    group_id=group_id,
                )

        return (
            PhaseResult(
                phase=self.name,
                status="success",
                items_processed=processed,
                items_affected=affected,
                duration_ms=(time.monotonic() - start) * 1000,
            ),
            [],
        )

    async def _expire_request(
        self,
        graph_store,
        request: dict,
        *,
        group_id: str,
    ) -> int:
        episode_rows = await graph_store.get_episode_evidence(
            request["episode_id"],
            group_id,
        )
        affected = 0
        for row in episode_rows:
            if row["evidence_id"] not in set(request.get("evidence_ids", [])):
                continue
            if row.get("status") not in {"pending", "deferred", "approved"}:
                continue
            await graph_store.update_evidence_status(
                row["evidence_id"],
                "expired",
                updates={"commit_reason": "adjudication_request_expired"},
                group_id=group_id,
            )
            affected += 1
        await graph_store.update_adjudication_request(
            request["request_id"],
            {"status": "expired", "resolved_at": utc_now_iso()},
            group_id,
        )
        return affected

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
