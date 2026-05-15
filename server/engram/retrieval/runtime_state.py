"""Runtime-state read model for REST, MCP, and epistemic evidence surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.utils.dates import utc_now_iso


async def build_runtime_state_surface(
    manager: Any,
    *,
    group_id: str,
    project_path: str | None = None,
) -> dict:
    """Read runtime state through the shared manager compatibility facade."""
    return await manager.get_runtime_state(
        group_id=group_id,
        project_path=project_path,
    )


class RuntimeStateService:
    """Build the shared runtime/config state exposed through GraphManager."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        runtime_mode: str,
        list_project_artifacts: Callable[..., Awaitable[list[Entity]]],
        artifact_is_stale: Callable[[Entity, int], bool],
        get_recall_metrics: Callable[[str], dict],
        get_epistemic_metrics: Callable[[str], dict],
    ) -> None:
        self._cfg = cfg
        self._runtime_mode = runtime_mode
        self._list_project_artifacts = list_project_artifacts
        self._artifact_is_stale = artifact_is_stale
        self._get_recall_metrics = get_recall_metrics
        self._get_epistemic_metrics = get_epistemic_metrics

    async def get_runtime_state(
        self,
        *,
        group_id: str = "default",
        project_path: str | None = None,
    ) -> dict:
        """Return effective runtime/config state plus artifact freshness."""
        artifacts = await self._list_project_artifacts(
            group_id=group_id,
            project_path=project_path,
        )
        stale_seconds = int(self._cfg.artifact_bootstrap_stale_seconds)
        stale_count = sum(
            1 for artifact in artifacts if self._artifact_is_stale(artifact, stale_seconds)
        )
        fresh_count = max(0, len(artifacts) - stale_count)
        last_observed = None
        for artifact in artifacts:
            observed = (artifact.attributes or {}).get("last_observed_at")
            if observed and (last_observed is None or observed > last_observed):
                last_observed = observed

        return {
            "projectName": Path(project_path).name if project_path else "Engram",
            "runtime": {
                "mode": self._runtime_mode,
            },
            "activation": {
                "consolidationProfile": self._cfg.consolidation_profile,
                "recallProfile": self._cfg.recall_profile,
                "integrationProfile": self._cfg.integration_profile,
            },
            "features": {
                "epistemicRoutingEnabled": self._cfg.epistemic_routing_enabled,
                "artifactBootstrapEnabled": self._cfg.artifact_bootstrap_enabled,
                "artifactRecallEnabled": self._cfg.artifact_recall_enabled,
                "runtimeExecutorEnabled": self._cfg.epistemic_runtime_executor_enabled,
                "decisionGraphEnabled": self._cfg.decision_graph_enabled,
                "epistemicReconcileEnabled": self._cfg.epistemic_reconcile_enabled,
                "answerContractEnabled": self._cfg.answer_contract_enabled,
                "claimStateModelingEnabled": self._cfg.claim_state_modeling_enabled,
                "recallNeedAnalyzerEnabled": self._cfg.recall_need_analyzer_enabled,
                "recallNeedGraphProbeEnabled": self._cfg.recall_need_graph_probe_enabled,
            },
            "artifactBootstrap": {
                "enabled": self._cfg.artifact_bootstrap_enabled,
                "projectPath": project_path,
                "artifactCount": len(artifacts),
                "freshArtifactCount": fresh_count,
                "staleArtifactCount": stale_count,
                "lastObservedAt": last_observed,
                "staleAfterSeconds": stale_seconds,
            },
            "stats": {
                "recallMetrics": self._get_recall_metrics(group_id),
                "epistemicMetrics": self._get_epistemic_metrics(group_id),
            },
            "generatedAt": utc_now_iso(),
        }
