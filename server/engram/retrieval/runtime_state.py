"""Runtime-state read model for REST, MCP, and epistemic evidence surfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from numbers import Number
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
        get_memory_operation_metrics: Callable[[str], dict],
        get_epistemic_metrics: Callable[[str], dict],
        get_packet_cache_summary: Callable[[str], dict] | None = None,
    ) -> None:
        self._cfg = cfg
        self._runtime_mode = runtime_mode
        self._list_project_artifacts = list_project_artifacts
        self._artifact_is_stale = artifact_is_stale
        self._get_recall_metrics = get_recall_metrics
        self._get_memory_operation_metrics = get_memory_operation_metrics
        self._get_epistemic_metrics = get_epistemic_metrics
        self._get_packet_cache_summary = get_packet_cache_summary or (lambda _group_id: {})

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

        artifact_bootstrap = {
            "enabled": self._cfg.artifact_bootstrap_enabled,
            "projectPath": project_path,
            "artifactCount": len(artifacts),
            "freshArtifactCount": fresh_count,
            "staleArtifactCount": stale_count,
            "lastObservedAt": last_observed,
            "staleAfterSeconds": stale_seconds,
        }
        recall_metrics = self._get_recall_metrics(group_id)
        memory_operation_metrics = self._get_memory_operation_metrics(group_id)
        epistemic_metrics = self._get_epistemic_metrics(group_id)
        packet_cache = self._get_packet_cache_summary(group_id)
        stats = {
            "recallMetrics": recall_metrics,
            "memoryOperationMetrics": memory_operation_metrics,
            "epistemicMetrics": epistemic_metrics,
            "packetCache": packet_cache,
        }

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
            "artifactBootstrap": artifact_bootstrap,
            "agentAdoption": _build_agent_adoption_guidance(
                artifact_bootstrap,
                recall_metrics=recall_metrics,
                epistemic_metrics=epistemic_metrics,
                project_path=project_path,
            ),
            "stats": stats,
            "generatedAt": utc_now_iso(),
        }


def _build_agent_adoption_guidance(
    artifact_bootstrap: Mapping[str, Any],
    *,
    recall_metrics: Mapping[str, Any],
    epistemic_metrics: Mapping[str, Any],
    project_path: str | None,
) -> dict[str, Any]:
    """Return compact adoption guidance for agents that only probe runtime state."""
    artifact_count = int(artifact_bootstrap.get("artifactCount") or 0)
    stale_count = int(artifact_bootstrap.get("staleArtifactCount") or 0)
    last_observed = artifact_bootstrap.get("lastObservedAt")
    artifact_gap = artifact_count == 0 or last_observed is None
    metrics_gap = _metrics_are_empty_or_zero(recall_metrics) and _metrics_are_empty_or_zero(
        epistemic_metrics
    )
    fresh_runtime = artifact_gap and metrics_gap
    bootstrap_enabled = bool(artifact_bootstrap.get("enabled", True))
    needs_bootstrap = bootstrap_enabled and (artifact_gap or stale_count > 0)
    can_bootstrap_now = bool(project_path and needs_bootstrap)

    required_tools = ["get_context"]
    if can_bootstrap_now:
        required_tools = ["claim_authority", "bootstrap_project", "get_context"]
    elif needs_bootstrap or fresh_runtime:
        required_tools = ["claim_authority", "get_context"]

    status = (
        "fresh_runtime"
        if fresh_runtime
        else "needs_project_bootstrap"
        if needs_bootstrap
        else "ready"
    )
    return {
        "status": status,
        "doNotTreatEmptyAsFailure": bool(fresh_runtime or needs_bootstrap),
        "requiredNextTools": required_tools,
        "beforeAnswer": {
            "required": True,
            "tools": required_tools,
            "reason": (
                "Follow these Engram tools before the next substantive answer; "
                "runtime-state metrics are diagnostics, not a substitute for "
                "authority, bootstrap, and context loading."
            ),
        },
        "emptyRuntimePolicy": (
            "Treat an empty or fresh Engram runtime as onboarding state. "
            "Bootstrap the project when possible before judging memory usefulness."
        ),
        "fileMemoryPolicy": (
            "Project-local memory files can supply visible scratch or conventions, "
            "but they do not replace Engram as the portable cross-context memory authority."
        ),
        "claimAuthority": {
            "tool": "claim_authority",
            "args": {
                "project_path": project_path or "<current_project_path>",
                "file_memory_present": "<true if local/file memory is visible>",
            },
            "reason": (
                "Ask Engram for the source-of-truth contract before choosing "
                "file-local memory over portable memory."
            ),
        },
        "bootstrap": {
            "tool": "bootstrap_project",
            "required": can_bootstrap_now,
            "args": {"project_path": project_path or "<current_project_path>"},
            "reason": (
                "A fresh or empty artifact substrate is onboarding state, not proof "
                "that Engram has no useful memory."
            ),
        },
        "reason": (
            "Connected but empty/fresh Engram runtime; call claim_authority, "
            "bootstrap the project when possible, then load context."
            if fresh_runtime
            else (
                "Project artifacts are missing or stale; refresh bootstrap before "
                "judging recall usefulness."
            )
            if needs_bootstrap
            else "Engram has runtime evidence available; load context and recall normally."
        ),
    }


def _metrics_are_empty_or_zero(metrics: Mapping[str, Any]) -> bool:
    values = list(_numeric_values(metrics))
    return not values or all(value == 0 for value in values)


def _numeric_values(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, Number):
        yield float(value)
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _numeric_values(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            yield from _numeric_values(nested)
