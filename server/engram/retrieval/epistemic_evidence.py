"""Epistemic evidence gathering and reconciliation service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from engram.config import ActivationConfig
from engram.models.epistemic import ArtifactHit, EpistemicBundle, EvidenceClaim
from engram.retrieval.epistemic import (
    apply_claim_states,
    build_memory_claims,
    build_runtime_claims,
    reconcile_claims,
    resolve_answer_contract,
    summarize_claim_states,
)


class EpistemicEvidenceService:
    """Gather planned evidence sources and reconcile claims for a routed turn."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        build_route: Callable[..., Awaitable[tuple]],
        bootstrap_project: Callable[..., Awaitable[dict]],
        recall: Callable[..., Awaitable[list[dict]]],
        search_artifacts: Callable[..., Awaitable[list[ArtifactHit]]],
        get_runtime_state: Callable[..., Awaitable[dict]],
        record_execution: Callable[..., None],
    ) -> None:
        self._cfg = cfg
        self._build_route = build_route
        self._bootstrap_project = bootstrap_project
        self._recall = recall
        self._search_artifacts = search_artifacts
        self._get_runtime_state = get_runtime_state
        self._record_execution = record_execution

    async def gather_epistemic_evidence(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need: Any = None,
    ) -> EpistemicBundle:
        """Route a question, gather planned evidence, and reconcile it."""
        frame, plan, routed_need, _initial_contract = await self._build_route(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )

        query_text = getattr(routed_need, "query_hint", None) or question
        memory_query = plan.source_queries.get("memory") or query_text
        artifact_query = plan.source_queries.get("artifacts") or query_text

        if plan.use_artifacts and project_path and self._cfg.artifact_bootstrap_enabled:
            await self._bootstrap_project(project_path, group_id=group_id)

        memory_results: list[dict] = []
        artifact_hits: list[ArtifactHit] = []
        runtime_state: dict | None = None

        if plan.use_memory:
            memory_results = await self._recall(
                query=memory_query,
                group_id=group_id,
                limit=max(1, plan.memory_budget),
                record_access=False,
            )
        if plan.use_artifacts:
            artifact_hits = await self._search_artifacts(
                query=artifact_query,
                group_id=group_id,
                project_path=project_path,
                limit=max(1, plan.artifact_budget),
            )
        if plan.use_runtime or plan.use_implementation:
            runtime_state = await self._get_runtime_state(
                group_id=group_id,
                project_path=project_path,
            )

        memory_claims = build_memory_claims(memory_results)
        artifact_claims = [claim for hit in artifact_hits for claim in hit.supporting_claims]
        runtime_claims = build_runtime_claims(runtime_state or {})
        implementation_claims: list[EvidenceClaim] = []
        all_claims = apply_claim_states(
            memory_claims + artifact_claims + runtime_claims + implementation_claims
        )
        claim_state_summary = summarize_claim_states(all_claims)
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=all_claims,
        )

        reconciliation = reconcile_claims(
            frame,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            answer_contract=answer_contract,
        )
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=all_claims,
            reconciliation=reconciliation,
        )
        reconciliation = reconcile_claims(
            frame,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            answer_contract=answer_contract,
        )

        artifact_stale_miss = bool(
            runtime_state
            and runtime_state.get("artifactBootstrap", {}).get("staleArtifactCount", 0)
            and not artifact_hits
        )
        self._record_execution(
            group_id,
            reconciliation,
            plan,
            answer_contract=answer_contract,
            artifact_stale_miss=artifact_stale_miss,
        )

        return EpistemicBundle(
            question_frame=frame,
            evidence_plan=plan,
            reconciliation=reconciliation,
            answer_contract=answer_contract,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            artifact_hits=artifact_hits,
            memory_results=memory_results,
            runtime_state=runtime_state,
            claim_state_summary=claim_state_summary,
        )
