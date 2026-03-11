"""Budgeted offline adjudication for unresolved ambiguous evidence."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, cast

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult
from engram.utils.dates import utc_now, utc_now_iso

logger = logging.getLogger(__name__)

_SUPPORTED_TAGS = {
    "coreference",
    "negation_scope",
    "temporal_attachment",
    "conflict_with_existing",
}
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_message_text(blocks: object) -> str:
    """Join Anthropic text blocks without assuming a specific response variant."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


class EdgeAdjudicationPhase(ConsolidationPhase):
    """Offline phase for expiring or optionally adjudicating v3 edge cases."""

    _daily_budget: dict[tuple[str, str], int] = {}

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
            limit=max(200, cfg.edge_adjudication_server_max_per_cycle * 5),
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

        budget_key = (group_id, now.date().isoformat())
        daily_used = self._daily_budget.get(budget_key, 0)
        cycle_used = 0

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
                continue

            if not cfg.edge_adjudication_server_enabled:
                continue
            if self._manager is None:
                return (
                    PhaseResult(
                        phase=self.name,
                        status="error",
                        error="graph_manager_required",
                        items_processed=processed,
                        items_affected=affected,
                        duration_ms=(time.monotonic() - start) * 1000,
                    ),
                    [],
                )
            if request.get("status") != "pending":
                continue
            if cycle_used >= cfg.edge_adjudication_server_max_per_cycle:
                continue
            if daily_used >= cfg.edge_adjudication_server_daily_budget:
                continue
            if age < timedelta(minutes=cfg.edge_adjudication_server_min_age_minutes):
                continue
            tags = set(request.get("ambiguity_tags", []))
            if not tags or not tags.issubset(_SUPPORTED_TAGS):
                continue

            try:
                resolution = await self._resolve_with_server(
                    request,
                    graph_store=graph_store,
                    group_id=group_id,
                    cfg=cfg,
                )
            except Exception as exc:
                logger.warning(
                    "edge_adjudication failed for request %s: %s",
                    request.get("request_id"),
                    exc,
                )
                await graph_store.update_adjudication_request(
                    request["request_id"],
                    {
                        "status": "error",
                        "resolution_source": "server_adjudication",
                        "attempt_count": int(request.get("attempt_count", 0) or 0) + 1,
                        "resolution_payload": {"error": str(exc)},
                    },
                    group_id,
                )
                continue

            if resolution is None:
                continue

            cycle_used += 1
            daily_used += 1
            if resolution.status in {"materialized", "rejected", "expired"}:
                affected += max(1, len(resolution.committed_ids))

        self._daily_budget[budget_key] = daily_used
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

    async def _resolve_with_server(
        self,
        request: dict,
        *,
        graph_store,
        group_id: str,
        cfg: ActivationConfig,
    ):
        manager = self._manager
        if manager is None:
            raise RuntimeError("graph_manager_required")
        episode_rows = await graph_store.get_episode_evidence(
            request["episode_id"],
            group_id,
        )
        candidate_evidence = [
            {
                "evidence_id": row["evidence_id"],
                "fact_class": row["fact_class"],
                "payload": row.get("payload", {}),
            }
            for row in episode_rows
            if row["evidence_id"] in set(request.get("evidence_ids", []))
        ]
        conflicting_facts: list[dict] = []
        if "conflict_with_existing" in set(request.get("ambiguity_tags", [])):
            conflicting_facts = [
                {
                    "fact_class": row["fact_class"],
                    "payload": row.get("payload", {}),
                    "status": row.get("status"),
                }
                for row in episode_rows
                if row["evidence_id"] not in set(request.get("evidence_ids", []))
                and row.get("status") == "committed"
            ][:5]

        resolution = await self._call_server_adjudicator(
            request,
            candidate_evidence=candidate_evidence,
            conflicting_facts=conflicting_facts,
            cfg=cfg,
        )
        if not resolution:
            return None
        return await manager.submit_adjudication_resolution(
            request["request_id"],
            entities=resolution.get("entities"),
            relationships=resolution.get("relationships"),
            reject_evidence_ids=resolution.get("reject_evidence_ids"),
            source="server_adjudication",
            model_tier="sonnet",
            rationale=resolution.get("rationale"),
            group_id=group_id,
        )

    async def _call_server_adjudicator(
        self,
        request: dict,
        *,
        candidate_evidence: list[dict],
        conflicting_facts: list[dict],
        cfg: ActivationConfig,
    ) -> dict | None:
        import anthropic

        prompt = {
            "request_id": request["request_id"],
            "ambiguity_tags": request.get("ambiguity_tags", []),
            "selected_text": request.get("selected_text", ""),
            "candidate_evidence": candidate_evidence,
            "conflicting_facts": conflicting_facts,
            "instructions": (
                "Return strict JSON with keys entities, relationships, "
                "reject_evidence_ids, rationale. Use empty lists when unsure. "
                "Do not include commentary outside JSON."
            ),
        }
        client = anthropic.Anthropic()

        def _create_message() -> Any:
            return client.messages.create(
                model=cfg.edge_adjudication_server_model,
                max_tokens=700,
                system=cast(
                    Any,
                    (
                        "You adjudicate structured memory edge cases. "
                        "Only resolve if highly confident. "
                        "Return JSON only."
                    ),
                ),
                messages=cast(Any, [{"role": "user", "content": json.dumps(prompt)}]),
            )

        response = await asyncio.to_thread(_create_message)
        text = _extract_message_text(response.content)
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ValueError("edge_adjudicator_no_json")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("edge_adjudicator_invalid_payload")
        return data

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
