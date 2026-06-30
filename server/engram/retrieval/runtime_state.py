"""Runtime-state read model for REST, MCP, and epistemic evidence surfaces."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable, Mapping
from numbers import Number
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.artifacts import _normalize_project_path
from engram.utils.dates import utc_now_iso

DEFAULT_RUNTIME_STATE_TIMEOUT_SECONDS = 1.0


async def build_runtime_state_surface(
    manager: Any,
    *,
    group_id: str,
    project_path: str | None = None,
    live: bool = False,
    timeout_seconds: float | None = None,
) -> dict:
    """Read runtime state through the shared manager compatibility facade."""
    return await manager.get_runtime_state(
        group_id=group_id,
        project_path=project_path,
        live=live,
        timeout_seconds=timeout_seconds,
    )


def build_fast_runtime_packet(
    cfg: ActivationConfig,
    *,
    runtime_mode: str,
    project_path: str | None = None,
    packet_cache_summary: Mapping[str, Any] | None = None,
) -> dict:
    """Return startup-safe runtime metadata without graph or artifact reads."""
    return {
        "projectName": Path(project_path).name if project_path else "Engram",
        "runtime": {
            "mode": runtime_mode,
            "surface": "fast_packet",
            "loadedGraphTouched": False,
        },
        "activation": {
            "consolidationProfile": cfg.consolidation_profile,
            "recallProfile": cfg.recall_profile,
            "integrationProfile": cfg.integration_profile,
        },
        "features": {
            "epistemicRoutingEnabled": cfg.epistemic_routing_enabled,
            "artifactBootstrapEnabled": cfg.artifact_bootstrap_enabled,
            "artifactRecallEnabled": cfg.artifact_recall_enabled,
            "runtimeExecutorEnabled": cfg.epistemic_runtime_executor_enabled,
            "decisionGraphEnabled": cfg.decision_graph_enabled,
            "epistemicReconcileEnabled": cfg.epistemic_reconcile_enabled,
            "answerContractEnabled": cfg.answer_contract_enabled,
            "claimStateModelingEnabled": cfg.claim_state_modeling_enabled,
            "recallNeedAnalyzerEnabled": cfg.recall_need_analyzer_enabled,
            "recallNeedGraphProbeEnabled": cfg.recall_need_graph_probe_enabled,
        },
        "artifactBootstrap": {
            "enabled": cfg.artifact_bootstrap_enabled,
            "projectPath": project_path,
            "artifactCount": 0,
            "freshArtifactCount": 0,
            "staleArtifactCount": 0,
            "lastObservedAt": None,
            "status": "not_inspected",
        },
        "agentAdoption": {
            "status": "startup_probe",
            "doNotTreatEmptyAsFailure": True,
            "requiredNextTools": ["get_context"],
            "beforeAnswer": {
                "required": False,
                "tools": ["get_context"],
                "reason": (
                    "Fast runtime packet intentionally skips graph and artifact inspection; "
                    "load context when prior memory could matter."
                ),
            },
            "reason": (
                "Startup-safe runtime metadata only; deep runtime and artifact state were not read."
            ),
        },
        "stats": {
            "source": "fast_runtime_packet",
            "packetCache": dict(packet_cache_summary or {}),
        },
        "generatedAt": utc_now_iso(),
    }


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
        self._cache: dict[tuple[str, str | None], tuple[dict[str, Any], float]] = {}

    async def get_runtime_state(
        self,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict:
        """Return effective runtime/config state plus artifact freshness.

        The default path is budgeted and cache-first so agent startup probes do
        not inherit deep artifact or graph-state latency. Operator callers can
        pass ``live=True`` for a fresh read.
        """
        cache_key = _runtime_cache_key(group_id, project_path)
        cached = self._cache.get(cache_key)
        if not live and cached is not None:
            return _with_cache_metadata(
                cached[0],
                live=False,
                cache_status="hit",
                cached_at=cached[1],
            )

        budget = (
            DEFAULT_RUNTIME_STATE_TIMEOUT_SECONDS
            if timeout_seconds is None and not live
            else timeout_seconds
        )
        try:
            if budget is None or budget <= 0:
                result = await self._build_live_runtime_state(
                    group_id=group_id,
                    project_path=project_path,
                )
            else:
                result = await asyncio.wait_for(
                    self._build_live_runtime_state(
                        group_id=group_id,
                        project_path=project_path,
                    ),
                    timeout=budget,
                )
        except TimeoutError:
            if cached is not None:
                return _with_cache_metadata(
                    cached[0],
                    live=live,
                    cache_status="stale_timeout",
                    cached_at=cached[1],
                    timeout=True,
                )
            result = build_fast_runtime_packet(
                self._cfg,
                runtime_mode=self._runtime_mode,
                project_path=project_path,
                packet_cache_summary=self._get_packet_cache_summary(group_id),
            )
            return _with_cache_metadata(
                result,
                live=live,
                cache_status="miss_timeout",
                cached_at=None,
                timeout=True,
            )

        cached_at = time.time()
        self._cache[cache_key] = (copy.deepcopy(result), cached_at)
        return _with_cache_metadata(
            result,
            live=True,
            cache_status="refreshed",
            cached_at=cached_at,
        )

    def invalidate_cache(
        self,
        *,
        group_id: str | None = None,
        project_path: str | None = None,
    ) -> None:
        """Drop cached runtime packets after bootstrap or other graph mutations."""
        if group_id is None and project_path is None:
            self._cache.clear()
            return
        keys_to_drop = [
            key
            for key in self._cache
            if (group_id is None or key[0] == group_id)
            and (
                project_path is None
                or key[1] == _runtime_cache_key(group_id or key[0], project_path)[1]
            )
        ]
        for key in keys_to_drop:
            del self._cache[key]

    async def _build_live_runtime_state(
        self,
        *,
        group_id: str = "default",
        project_path: str | None = None,
    ) -> dict:
        """Build runtime/config state from live artifact and metric sources."""
        started = time.perf_counter()
        artifact_started = time.perf_counter()
        artifacts = await self._list_project_artifacts(
            group_id=group_id,
            project_path=project_path,
        )
        artifact_ms = _elapsed_ms(artifact_started)
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

        metrics_started = time.perf_counter()
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
        metrics_ms = _elapsed_ms(metrics_started)
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
            "diagnostics": {
                "live": True,
                "cacheStatus": "live",
                "runtimeStateAgeSeconds": 0,
                "timeout": False,
                "stageTimingsMs": {
                    "runtime_state": _elapsed_ms(started),
                    "artifact_bootstrap": artifact_ms,
                    "runtime_metrics": metrics_ms,
                },
            },
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


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _runtime_cache_key(group_id: str, project_path: str | None) -> tuple[str, str | None]:
    return group_id, _normalize_project_path(project_path)


def _with_cache_metadata(
    payload: dict[str, Any],
    *,
    live: bool,
    cache_status: str,
    cached_at: float | None,
    timeout: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    now = time.time()
    diagnostics = dict(result.get("diagnostics") or {})
    stage_timings = dict(diagnostics.get("stageTimingsMs") or {})
    diagnostics.update(
        {
            "live": live,
            "cacheStatus": cache_status,
            "runtimeStateAgeSeconds": 0
            if cached_at is None
            else max(0, int(now - cached_at)),
            "timeout": timeout,
            "stageTimingsMs": stage_timings,
        }
    )
    result["diagnostics"] = diagnostics
    return result
