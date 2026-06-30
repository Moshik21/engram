"""Payload shaping for AXI commands."""

from __future__ import annotations

import shlex
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram.axi.budgets import compact_whitespace, first_present, truncate_text
from engram.axi.client import AxiRestClient, AxiRestError
from engram.harness_adoption import runtime_needs_bootstrap

LIFECYCLE = "Capture -> Cue -> Project -> Recall -> Consolidate"
DESCRIPTION = "Long-term memory brain for AI agents"
HOME_PROBE_TIMEOUT_SECONDS = 2.5
HOME_BOOTSTRAP_TIMEOUT_SECONDS = 55.0
FOLLOWUP_TIMEOUT_SECONDS = 10.0
VALUE_TIMEOUT_SECONDS = 20.0


@dataclass
class AxiResult:
    payload: dict[str, Any]
    exit_code: int = 0


def build_home_payload(
    client: AxiRestClient,
    *,
    project_path: str | None,
    topic_hint: str | None,
    budget: int,
    trace_file: str | None = None,
    trace_client: str | None = None,
    followup_trace_origin: str | None = None,
) -> AxiResult:
    """Build the default read-only AXI home packet."""
    base: dict[str, Any] = {
        "bin": _engram_bin(),
        "description": DESCRIPTION,
        "server": client.server_url,
    }
    probe_client = _home_probe_client(client)
    probes = _home_probe_payloads(probe_client, project_path=project_path)
    health = probes["health"]
    runtime = probes["runtime"]
    storage = probes["storage"]
    bootstrap_summary = _maybe_auto_bootstrap_home(
        probe_client,
        project_path=project_path,
        runtime=runtime,
        probes=probes,
    )
    runtime = probes["runtime"]
    health_error = probes.get("health_error")
    runtime_error = probes.get("runtime_error")
    storage_error = probes.get("storage_error")
    any_probe_succeeded = any(
        isinstance(item, dict) and item.get("status") != "degraded"
        for item in (health, runtime, storage)
    )
    if health_error and not _is_timeout_error(health_error) and not any_probe_succeeded:
        return AxiResult(
            payload={
                **base,
                "status": "offline",
                "error": health_error.message,
                "install": _local_install_state(),
                "next": _home_unavailable_next_actions(timed_out=False),
            }
        )

    if not isinstance(health, dict):
        return AxiResult(
            payload={
            **base,
            "status": "offline",
            "error": "Malformed health response",
            "install": _local_install_state(),
            "next": _home_unavailable_next_actions(timed_out=False),
            }
        )
    status = str(health.get("status") or "healthy")
    payload = {
        **base,
        "status": status,
        "mode": _mode_from_payloads(health, runtime),
        "transport": _transport_from_storage(storage),
        "storage": _compact_storage(storage),
        "brain": _compact_brain(runtime, project_path=project_path),
        "context": _context_pointer(
            runtime,
            topic_hint=topic_hint,
            project_path=project_path,
            server_url=client.server_url,
            timeout_seconds=getattr(client, "timeout_seconds", None),
            trace_file=trace_file,
            trace_client=trace_client,
            followup_trace_origin=followup_trace_origin,
        ),
        "next": _home_next_actions(
            project_path=project_path,
            server_url=client.server_url,
            timeout_seconds=getattr(client, "timeout_seconds", None),
            trace_file=trace_file,
            trace_client=trace_client,
            followup_trace_origin=followup_trace_origin,
        ),
    }
    if bootstrap_summary is not None:
        payload["bootstrap"] = bootstrap_summary
        observed = int(bootstrap_summary.get("observed") or 0)
        if observed > 0 and bootstrap_summary.get("status") != "error":
            brain = payload.get("brain") or {}
            brain["artifact_count"] = max(int(brain.get("artifact_count") or 0), observed)
            brain["artifact_status"] = "bootstrapped"
            payload["brain"] = brain
    if health.get("error"):
        payload["error"] = health.get("error")
    elif runtime_error and storage_error and not any_probe_succeeded:
        payload["error"] = runtime_error.message
    return AxiResult(payload=payload, exit_code=0)


def build_context_payload(
    client: AxiRestClient,
    *,
    topic_hint: str | None,
    project_path: str | None,
    budget: int,
    full: bool = False,
) -> AxiResult:
    try:
        context = client.context(
            max_tokens=budget,
            topic_hint=topic_hint,
            project_path=project_path,
        )
    except AxiRestError as exc:
        return _error_result("context", exc)
    text = str(context.get("context") or "")
    rendered, truncated, original_len = truncate_text(
        text,
        budget_tokens=None if full else budget,
    )
    payload = {
        "operation": "context",
        "status": context.get("status") or "ok",
        "format": context.get("format") or "structured",
        "entity_count": first_present(context, "entityCount", "entity_count") or 0,
        "fact_count": first_present(context, "factCount", "fact_count") or 0,
        "token_estimate": first_present(context, "tokenEstimate", "token_estimate") or 0,
        "truncated": truncated,
        "original_chars": original_len,
        "context": rendered,
    }
    packet_cache = context.get("packet_cache") or context.get("packetCache")
    cached_packets = context.get("cached_packets") or context.get("cachedPackets") or []
    if isinstance(packet_cache, dict):
        payload["packet_cache"] = {
            "hit": bool(packet_cache.get("hit")),
            "packet_count": packet_cache.get("packet_count")
            or packet_cache.get("packetCount")
            or 0,
            "scopes": packet_cache.get("scopes") or {},
        }
    if isinstance(cached_packets, list) and cached_packets:
        payload["packets"] = _compact_context_packets(cached_packets, budget=budget)
    budget_payload = context.get("budget")
    if isinstance(budget_payload, dict):
        payload["budget"] = budget_payload
    lifecycle = context.get("lifecycle")
    if isinstance(lifecycle, dict) and lifecycle:
        payload["lifecycle"] = lifecycle
    diagnostics = context.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        payload["diagnostics"] = diagnostics
    if truncated:
        payload["next"] = [
            {
                "cmd": "engram axi context --full",
                "reason": "Show untruncated context",
            }
        ]
    return AxiResult(payload=payload)


def build_recall_payload(
    client: AxiRestClient,
    *,
    query: str,
    limit: int,
    budget: int,
    project_path: str | None = None,
    full: bool = False,
) -> AxiResult:
    try:
        recall = client.recall(query, limit=limit, project_path=project_path)
    except AxiRestError as exc:
        return _error_result("recall", exc)

    results = recall.get("results") or recall.get("items") or []
    if isinstance(results, list):
        display_results = results[:limit]
    else:
        display_results = []
    compact_results = _compact_recall_results(display_results, budget=budget, full=full)
    packets = recall.get("packets") if isinstance(recall.get("packets"), list) else []
    lifecycle = recall.get("lifecycle") or {}
    payload = {
        "operation": "recall",
        "status": recall.get("status") or "ok",
        "query": query,
        "result_count": len(compact_results),
        "packet_count": len(packets),
        "total_result_count": lifecycle.get("resultCount")
        or (len(results) if isinstance(results, list) else 0),
        "results": compact_results,
    }
    if packets:
        payload["packets"] = _compact_context_packets(packets, budget=budget)
    if isinstance(lifecycle, dict) and lifecycle:
        payload["lifecycle"] = lifecycle
    budget_payload = recall.get("budget")
    if isinstance(budget_payload, dict):
        payload["budget"] = budget_payload
    diagnostics = recall.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        payload["diagnostics"] = diagnostics
    return AxiResult(payload=payload)


def build_value_payload(client: AxiRestClient) -> AxiResult:
    """Build a compact memory value report for agent startup/follow-up."""
    try:
        report = client.evaluation_report(live_cost=True)
    except AxiRestError as exc:
        return _error_result("value", exc)

    memory_value = report.get("memory_value") or report.get("memoryValue") or {}
    cost = memory_value.get("cost") or {}
    benefit = memory_value.get("benefit") or {}
    payload = {
        "operation": "value",
        "status": memory_value.get("status") or "needs_samples",
        "cost": _compact_value_cost(cost),
        "benefit": {
            "memory_need_precision": (
                benefit.get("memory_need_precision")
                or benefit.get("memoryNeedPrecision")
                or 0
            ),
            "useful_packet_rate": (
                benefit.get("useful_packet_rate") or benefit.get("usefulPacketRate") or 0
            ),
            "continuity_lift": (
                benefit.get("session_continuity_lift")
                or benefit.get("sessionContinuityLift")
                or 0
            ),
        },
        "next": [
            {
                "cmd": (
                    'engram axi context --project "$PWD" --budget 800 '
                    f"--timeout {FOLLOWUP_TIMEOUT_SECONDS:g}"
                ),
                "reason": "Use cached/project context when prior memory matters",
            },
            {
                "cmd": (
                    'engram axi recall "query" --limit 5 '
                    f"--timeout {FOLLOWUP_TIMEOUT_SECONDS:g}"
                ),
                "reason": "Use explicit recall for long-tail memory searches",
            },
        ],
    }
    return AxiResult(payload=payload)


def _compact_value_cost(cost: Any) -> dict[str, Any]:
    if not isinstance(cost, dict):
        cost = {}
    payload = {
        "source": "live_runtime",
        "operation_count": _value_int(cost, "operation_count", "operationCount"),
        "p95_added_latency_ms": _value_float(
            cost,
            "p95_added_latency_ms",
            "p95AddedLatencyMs",
        ),
        "cache_hit_rate": _value_float(cost, "cache_hit_rate", "cacheHitRate"),
        "budget_miss_rate": _value_float(
            cost,
            "budget_miss_rate",
            "budgetMissRate",
        ),
        "skipped_count": _value_int(cost, "skipped_count", "skippedCount"),
    }
    recent_problem_samples = _value_list(
        cost,
        "recent_problem_samples",
        "recentProblemSamples",
    )
    if recent_problem_samples:
        payload["recent_problem_samples"] = recent_problem_samples
    modes = _compact_value_modes(cost)
    if modes:
        payload["read_path"] = _value_path_cost(modes, _is_value_read_mode)
        payload["write_path"] = _value_path_cost(modes, _is_value_write_mode)
        payload["top_modes_by_p95"] = sorted(
            modes.values(),
            key=lambda mode: mode.get("p95_added_latency_ms", 0),
            reverse=True,
        )[:5]
        payload["mode_breakdown"] = modes
    return payload


def _compact_value_modes(cost: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_modes = cost.get("by_mode") or cost.get("byMode") or cost.get("modes") or {}
    if not isinstance(raw_modes, dict):
        return {}
    modes: dict[str, dict[str, Any]] = {}
    for mode, mode_cost in raw_modes.items():
        if not isinstance(mode_cost, dict):
            continue
        label = str(mode)
        modes[label] = {
            "mode": label,
            "operation_count": _value_int(
                mode_cost,
                "operation_count",
                "operationCount",
            ),
            "p95_added_latency_ms": _value_float(
                mode_cost,
                "p95_added_latency_ms",
                "p95AddedLatencyMs",
            ),
            "avg_added_latency_ms": _value_float(
                mode_cost,
                "avg_added_latency_ms",
                "avgAddedLatencyMs",
            ),
            "timeout_count": _value_int(mode_cost, "timeout_count", "timeoutCount"),
            "degraded_count": _value_int(mode_cost, "degraded_count", "degradedCount"),
            "budget_miss_count": _value_int(
                mode_cost,
                "budget_miss_count",
                "budgetMissCount",
            ),
            "cache_hit_count": _value_int(
                mode_cost,
                "cache_hit_count",
                "cacheHitCount",
            ),
            "cache_miss_count": _value_int(
                mode_cost,
                "cache_miss_count",
                "cacheMissCount",
            ),
            "skipped_count": _value_int(mode_cost, "skipped_count", "skippedCount"),
            "status_counts": _value_counts(mode_cost, "status_counts", "statusCounts"),
            "skip_reason_counts": _value_counts(
                mode_cost,
                "skip_reason_counts",
                "skipReasonCounts",
            ),
            "operation_counts": _value_counts(
                mode_cost,
                "operation_counts",
                "operationCounts",
            ),
            "source_counts": _value_counts(mode_cost, "source_counts", "sourceCounts"),
        }
    return modes


def _value_path_cost(
    modes: dict[str, dict[str, Any]],
    predicate: Callable[[str], bool],
) -> dict[str, Any]:
    selected = [mode for name, mode in modes.items() if predicate(name)]
    operation_count = sum(int(mode.get("operation_count") or 0) for mode in selected)
    timeout_count = sum(int(mode.get("timeout_count") or 0) for mode in selected)
    degraded_count = sum(int(mode.get("degraded_count") or 0) for mode in selected)
    budget_miss_count = sum(int(mode.get("budget_miss_count") or 0) for mode in selected)
    cache_hit_count = sum(int(mode.get("cache_hit_count") or 0) for mode in selected)
    cache_miss_count = sum(int(mode.get("cache_miss_count") or 0) for mode in selected)
    skipped_count = sum(int(mode.get("skipped_count") or 0) for mode in selected)
    cache_total = cache_hit_count + cache_miss_count
    return {
        "status": "measured" if operation_count else "needs_samples",
        "operation_count": operation_count,
        "p95_added_latency_ms": max(
            (float(mode.get("p95_added_latency_ms") or 0) for mode in selected),
            default=0,
        ),
        "timeout_rate": _value_rate(timeout_count, operation_count),
        "degraded_rate": _value_rate(degraded_count, operation_count),
        "budget_miss_rate": _value_rate(budget_miss_count, operation_count),
        "cache_hit_rate": _value_rate(cache_hit_count, cache_total),
        "skipped_count": skipped_count,
    }


def _is_value_read_mode(mode: str) -> bool:
    normalized = mode.lower()
    if _is_value_write_mode(normalized):
        return False
    return (
        "recall" in normalized
        or "context" in normalized
        or "packet" in normalized
        or normalized in {"lite", "medium", "deep", "full"}
    )


def _is_value_write_mode(mode: str) -> bool:
    normalized = mode.lower()
    return any(marker in normalized for marker in ("observe", "remember", "capture"))


def _value_int(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                continue
    return 0


def _value_float(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0


def _value_counts(mapping: dict[str, Any], *keys: str) -> dict[str, int]:
    for key in keys:
        value = mapping.get(key)
        if not isinstance(value, dict):
            continue
        counts: dict[str, int] = {}
        for item_key, item_value in value.items():
            count = _value_int({"value": item_value}, "value")
            if count:
                counts[str(item_key)] = count
        return counts
    return {}


def _value_list(mapping: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _value_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0
    return round(numerator / denominator, 4)


def build_packet_cache_clear_payload(client: AxiRestClient) -> AxiResult:
    """Build the manual packet-cache clear command payload."""
    try:
        result = client.clear_packet_cache()
    except AxiRestError as exc:
        return _error_result("packet-cache.clear", exc)

    packet_cache = result.get("packetCache") or result.get("packet_cache") or {}
    payload = {
        "operation": "packet-cache.clear",
        "status": result.get("status") or "cleared",
        "cleared_count": result.get("clearedCount") or result.get("cleared_count") or 0,
        "packet_cache": {
            "entry_count": packet_cache.get("entryCount")
            or packet_cache.get("entry_count")
            or 0,
            "fresh_count": packet_cache.get("freshCount")
            or packet_cache.get("fresh_count")
            or 0,
            "hit_count": packet_cache.get("hitCount") or packet_cache.get("hit_count") or 0,
            "persistent": bool(packet_cache.get("persistent")),
            "path": packet_cache.get("path"),
        },
        "next": [
            {
                "cmd": (
                    'engram axi context --project "$PWD" '
                    f"--timeout {FOLLOWUP_TIMEOUT_SECONDS:g}"
                ),
                "reason": "Rebuild cached project context on demand",
            }
        ],
    }
    return AxiResult(payload=payload)


def build_packet_cache_summary_payload(client: AxiRestClient) -> AxiResult:
    """Build the read-only packet-cache summary command payload."""
    try:
        result = client.packet_cache()
    except AxiRestError as exc:
        return _error_result("packet-cache", exc)

    packet_cache = result.get("packetCache") or result.get("packet_cache") or {}
    payload = {
        "operation": "packet-cache.summary",
        "status": result.get("status") or "ok",
        "packet_cache": {
            "entry_count": packet_cache.get("entryCount")
            or packet_cache.get("entry_count")
            or 0,
            "fresh_count": packet_cache.get("freshCount")
            or packet_cache.get("fresh_count")
            or 0,
            "invalidated_count": packet_cache.get("invalidatedCount")
            or packet_cache.get("invalidated_count")
            or 0,
            "expired_count": packet_cache.get("expiredCount")
            or packet_cache.get("expired_count")
            or 0,
            "hit_count": packet_cache.get("hitCount") or packet_cache.get("hit_count") or 0,
            "scopes": packet_cache.get("scopes") or {},
            "persistent": bool(packet_cache.get("persistent")),
            "path": packet_cache.get("path"),
        },
        "next": [
            {
                "cmd": (
                    'engram axi context --project "$PWD" '
                    f"--timeout {FOLLOWUP_TIMEOUT_SECONDS:g}"
                ),
                "reason": "Warm or reuse project context packets",
            },
            {
                "cmd": "engram axi packet-cache clear",
                "reason": "Clear stale packets without deleting memories",
            },
        ],
    }
    return AxiResult(payload=payload)


def build_storage_payload(client: AxiRestClient) -> AxiResult:
    try:
        storage = client.storage(live=True, timeout_seconds=FOLLOWUP_TIMEOUT_SECONDS)
    except AxiRestError as exc:
        return _error_result("storage", exc)
    payload = {
        "operation": "storage",
        "status": "ok",
        **_compact_storage(storage, include_paths=True),
    }
    return AxiResult(payload=payload)


def build_doctor_payload(client: AxiRestClient, *, project_path: str | None = None) -> AxiResult:
    checks: list[dict[str, Any]] = []
    for name, call in (
        ("health", client.health),
        ("runtime", lambda: client.runtime(project_path=project_path)),
        ("storage", client.storage),
    ):
        try:
            call()
            checks.append({"name": name, "status": "pass"})
        except AxiRestError as exc:
            checks.append({"name": name, "status": "fail", "detail": exc.message})
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return AxiResult(
        payload={
            "operation": "doctor",
            "status": status,
            "server": client.server_url,
            "checks": checks,
        },
        exit_code=0 if status == "pass" else 1,
    )


def build_write_payload(
    client: AxiRestClient,
    *,
    operation: str,
    content: str,
    source: str,
    conversation_date: str | None,
) -> AxiResult:
    if not content.strip():
        return AxiResult(
            payload={
                "operation": operation,
                "status": "error",
                "error": "No input received on stdin",
                "next": [
                    {
                        "cmd": f"printf '%s' 'memory text' | engram axi {operation} --stdin",
                        "reason": "Send explicit content to capture",
                    }
                ],
            },
            exit_code=2,
        )
    try:
        if operation == "observe":
            result = client.observe(
                content=content,
                source=source,
                conversation_date=conversation_date,
            )
        elif operation == "remember":
            result = client.remember(
                content=content,
                source=source,
                conversation_date=conversation_date,
            )
        else:
            raise ValueError(f"Unsupported write operation: {operation}")
    except AxiRestError as exc:
        return _error_result(operation, exc)

    return AxiResult(payload=_compact_write_result(operation, result))


def build_bootstrap_payload(
    client: AxiRestClient,
    *,
    project_path: str,
    include_patterns: list[str] | None = None,
) -> AxiResult:
    try:
        result = client.bootstrap(
            project_path=project_path,
            include_patterns=include_patterns,
        )
    except AxiRestError as exc:
        return _error_result("bootstrap", exc)
    payload = {
        "operation": "bootstrap",
        "status": result.get("status") or "ok",
        "project": project_path,
        "observed": _bootstrap_observed_count(result),
        "skipped": first_present(result, "skipped", "skippedCount") or 0,
    }
    message = result.get("message")
    if message:
        payload["message"] = message
    return AxiResult(payload=payload)


def _maybe_auto_bootstrap_home(
    client: AxiRestClient,
    *,
    project_path: str | None,
    runtime: dict[str, Any],
    probes: dict[str, Any],
) -> dict[str, Any] | None:
    if not project_path or not isinstance(runtime, dict):
        return None
    if not runtime_needs_bootstrap(runtime):
        return None
    bootstrap_client = client
    with_timeout = getattr(client, "with_timeout", None)
    if callable(with_timeout):
        bootstrap_client = with_timeout(HOME_BOOTSTRAP_TIMEOUT_SECONDS)
    try:
        result = bootstrap_client.bootstrap(project_path=project_path)
    except AxiRestError as exc:
        return {
            "status": "error",
            "project": project_path,
            "error": exc.message,
            "auto": True,
        }
    probes["runtime"] = _home_runtime_packet_live(
        bootstrap_client,
        project_path=project_path,
    )
    return {
        "status": result.get("status") or "ok",
        "project": project_path,
        "observed": _bootstrap_observed_count(result),
        "skipped": first_present(result, "skipped", "skippedCount") or 0,
        "auto": True,
    }


def _bootstrap_observed_count(result: dict[str, Any]) -> int:
    raw = first_present(
        result,
        "files_observed",
        "observed",
        "observedCount",
        "count",
    )
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, int):
        return raw
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _home_probe_payloads(
    client: AxiRestClient,
    *,
    project_path: str | None,
) -> dict[str, Any]:
    calls = {
        "health": client.health,
        "runtime": lambda: _home_runtime_packet(client, project_path=project_path),
        "storage": client.storage,
    }
    payloads: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(calls), thread_name_prefix="engram-axi-home") as pool:
        futures = {pool.submit(call): name for name, call in calls.items()}
        for future, name in futures.items():
            try:
                payload = future.result()
                payloads[name] = payload if isinstance(payload, dict) else {}
            except AxiRestError as exc:
                payloads[name] = {"status": "degraded", "error": exc.message}
                payloads[f"{name}_error"] = exc
    return payloads


def _home_runtime_packet(client: AxiRestClient, *, project_path: str | None) -> dict[str, Any]:
    runtime_fast = getattr(client, "runtime_fast", None)
    if callable(runtime_fast):
        return runtime_fast(project_path=project_path)
    return client.runtime(project_path=project_path)


def _home_runtime_packet_live(
    client: AxiRestClient,
    *,
    project_path: str | None,
) -> dict[str, Any]:
    runtime = getattr(client, "runtime", None)
    if callable(runtime):
        try:
            return runtime(
                project_path=project_path,
                live=True,
                timeout_seconds=10.0,
            )
        except TypeError:
            return runtime(project_path=project_path)
    return _home_runtime_packet(client, project_path=project_path)


def _is_timeout_error(exc: AxiRestError) -> bool:
    return "timed out" in exc.message.lower()


def _home_probe_client(client: AxiRestClient) -> AxiRestClient:
    timeout = min(
        float(getattr(client, "timeout_seconds", HOME_PROBE_TIMEOUT_SECONDS)),
        HOME_PROBE_TIMEOUT_SECONDS,
    )
    with_timeout = getattr(client, "with_timeout", None)
    if callable(with_timeout):
        return with_timeout(timeout)
    return client


def _error_result(operation: str, exc: AxiRestError) -> AxiResult:
    next_actions = [
        {"cmd": "engramctl status", "reason": "Inspect local runtime status"},
        {"cmd": "engramctl start", "reason": "Start the local Engram runtime"},
    ]
    if operation == "value" and _is_timeout_error(exc):
        next_actions = [
            {
                "cmd": f"engram axi value --timeout {VALUE_TIMEOUT_SECONDS:g}",
                "reason": "Retry the value report with the native-store report budget",
            },
            {"cmd": "engramctl status", "reason": "Inspect local runtime status"},
        ]
    return AxiResult(
        payload={
            "operation": operation,
            "status": "error",
            "server": exc.url,
            "error": exc.message,
            "next": next_actions,
        },
        exit_code=1,
    )


def _engram_bin() -> str:
    return shutil.which("engram") or Path(sys.argv[0]).name or "engram"


def _local_install_state() -> dict[str, Any]:
    home = Path.home()
    lite_env = home / ".engram" / ".env"
    full_env = home / ".engram" / "full" / ".env"
    env_path = lite_env if lite_env.exists() else full_env if full_env.exists() else None
    state = {"home": str(home / ".engram")}
    if env_path is None:
        state["variant"] = "not_found"
        return state
    parsed = _parse_env_file(env_path)
    state["config"] = str(env_path)
    state["variant"] = parsed.get("ENGRAM_INSTALL_VARIANT") or "local"
    mode = parsed.get("ENGRAM_MODE")
    if mode:
        state["mode"] = mode
    return state


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _home_unavailable_next_actions(*, timed_out: bool) -> list[dict[str, str]]:
    if timed_out:
        return [
            {
                "cmd": "engramctl status",
                "reason": "Inspect whether the local runtime is busy or degraded",
            },
            {
                "cmd": f"engram axi --timeout {FOLLOWUP_TIMEOUT_SECONDS:g}",
                "reason": "Retry the home packet with a longer probe budget",
            },
        ]
    return [
        {"cmd": "engramctl start", "reason": "Start the local Engram runtime"},
        {"cmd": "engramctl status", "reason": "Inspect mode, paths, and logs"},
    ]


def _mode_from_payloads(health: dict[str, Any], runtime: dict[str, Any]) -> str:
    mode = health.get("mode")
    if mode:
        return str(mode)
    runtime_mode = (runtime.get("runtime") or {}).get("mode")
    return str(runtime_mode or "unknown")


def _transport_from_storage(storage: dict[str, Any]) -> str:
    backend = storage.get("backend")
    if backend == "helix_native":
        return "native"
    if backend:
        return str(backend)
    return "unknown"


def _compact_storage(storage: dict[str, Any], *, include_paths: bool = False) -> dict[str, Any]:
    if storage.get("status") == "degraded":
        return {"status": "degraded", "error": storage.get("error")}
    disk = storage.get("disk") or {}
    counts = storage.get("counts") or {}
    result: dict[str, Any] = {
        "mode": storage.get("mode") or "unknown",
        "backend": storage.get("backend") or "unknown",
        "size": disk.get("humanSize") or "unknown",
        "counts": {
            "episodes": counts.get("episodes", 0),
            "entities": counts.get("entities", 0),
            "relationships": counts.get("relationships", 0),
            "cues": counts.get("cues", 0),
        },
    }
    if include_paths:
        paths = storage.get("paths") or []
        result["paths"] = [
            {
                "label": item.get("label"),
                "path": item.get("path"),
                "exists": item.get("exists"),
                "size": item.get("humanSize"),
            }
            for item in paths
            if isinstance(item, dict)
        ]
    else:
        data_path = _primary_storage_path(storage)
        if data_path:
            result["data_dir"] = data_path
    return result


def _primary_storage_path(storage: dict[str, Any]) -> str | None:
    for item in storage.get("paths") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").lower()
        if "helix native" in label or "sqlite database" in label:
            return str(item.get("path") or "")
    return None


def _compact_brain(runtime: dict[str, Any], *, project_path: str | None) -> dict[str, Any]:
    adoption = runtime.get("agentAdoption") or {}
    artifact = runtime.get("artifactBootstrap") or {}
    packet_cache = (runtime.get("stats") or {}).get("packetCache") or {}
    fresh_packets = int(packet_cache.get("fresh_count") or packet_cache.get("freshCount") or 0)
    return {
        "lifecycle": LIFECYCLE,
        "project": project_path or artifact.get("projectPath") or runtime.get("projectName"),
        "artifact_status": adoption.get("status") or "unknown",
        "artifact_count": artifact.get("artifactCount", 0),
        "packet_cache": {
            "status": "warm" if fresh_packets > 0 else "cold",
            "fresh": fresh_packets,
            "hits": packet_cache.get("hit_count") or packet_cache.get("hitCount") or 0,
        },
        "required_next_tools": adoption.get("requiredNextTools") or [],
    }


def _compact_context(
    context: dict[str, Any],
    *,
    project_path: str | None,
    budget: int,
) -> dict[str, Any]:
    if context.get("status") == "degraded":
        return {"status": "degraded", "error": context.get("error")}
    text = context.get("context") or ""
    summary, truncated, original_len = truncate_text(
        compact_whitespace(text),
        budget_tokens=min(budget, 120),
        minimum_chars=240,
    )
    return {
        "status": "ready" if text else "empty",
        "project": project_path,
        "summary": summary,
        "truncated": truncated,
        "original_chars": original_len,
    }


def _compact_context_packets(
    packets: list[Any],
    *,
    budget: int,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    per_packet_budget = max(40, min(120, budget // max(1, min(len(packets), 5))))
    for packet in packets[:5]:
        if not isinstance(packet, dict):
            continue
        trust = packet.get("trust") if isinstance(packet.get("trust"), dict) else {}
        summary, truncated, original_len = truncate_text(
            compact_whitespace(str(packet.get("summary") or "")),
            budget_tokens=per_packet_budget,
            minimum_chars=120,
        )
        compact.append(
            {
                "type": packet.get("packet_type") or packet.get("packetType") or "memory",
                "title": packet.get("title") or "Memory",
                "summary": summary,
                "truncated": truncated,
                "original_chars": original_len,
                "trust": {
                    "freshness": trust.get("freshness") or "unknown",
                    "source": trust.get("source") or "cache",
                    "confidence": trust.get("confidence") or packet.get("confidence") or 0,
                    "why": trust.get("why_now")
                    or trust.get("whyNow")
                    or packet.get("why_now")
                    or packet.get("whyNow")
                    or "",
                    "provenance": trust.get("provenance_count")
                    or trust.get("provenanceCount")
                    or len(packet.get("provenance") or []),
                    "evidence": trust.get("evidence_count")
                    or trust.get("evidenceCount")
                    or len(packet.get("evidence_lines") or packet.get("evidenceLines") or []),
                    "belief": trust.get("belief_status")
                    or trust.get("beliefStatus")
                    or "unknown",
                    "confirmed": trust.get("confirmed_count")
                    or trust.get("confirmedCount")
                    or 0,
                    "corrected": trust.get("corrected_count")
                    or trust.get("correctedCount")
                    or 0,
                    "dismissed": trust.get("dismissed_count")
                    or trust.get("dismissedCount")
                    or 0,
                    "last_confirmed": trust.get("last_confirmed_at")
                    or trust.get("lastConfirmedAt"),
                    "last_corrected": trust.get("last_corrected_at")
                    or trust.get("lastCorrectedAt"),
                },
            }
        )
    return compact


def _context_pointer(
    runtime: dict[str, Any],
    *,
    topic_hint: str | None,
    project_path: str | None,
    server_url: str | None = None,
    timeout_seconds: float | None = None,
    trace_file: str | None = None,
    trace_client: str | None = None,
    followup_trace_origin: str | None = None,
) -> dict[str, Any]:
    if runtime.get("status") == "degraded":
        return {"status": "unknown", "error": runtime.get("error")}
    adoption = runtime.get("agentAdoption") or {}
    command = "engram axi context --budget 800"
    if project_path:
        command += ' --project "$PWD"'
    if topic_hint:
        command += f" --topic {topic_hint!r}"
    followup_flags = _followup_command_flags(
        server_url=server_url,
        timeout_seconds=timeout_seconds,
        trace_file=trace_file,
        trace_client=trace_client,
        followup_trace_origin=followup_trace_origin,
    )
    if followup_flags:
        command += f" {followup_flags}"
    return {
        "status": "available",
        "project": project_path or runtime.get("projectName"),
        "summary": adoption.get("reason")
        or "Load compact Engram context when prior memory matters.",
        "cmd": command,
    }


def _home_next_actions(
    *,
    project_path: str | None,
    server_url: str | None = None,
    timeout_seconds: float | None = None,
    trace_file: str | None = None,
    trace_client: str | None = None,
    followup_trace_origin: str | None = None,
) -> list[dict[str, str]]:
    project_flag = ' --project "$PWD"' if project_path else ""
    followup_flags = _followup_command_flags(
        server_url=server_url,
        timeout_seconds=timeout_seconds,
        trace_file=trace_file,
        trace_client=trace_client,
        followup_trace_origin=followup_trace_origin,
    )
    followup_suffix = f" {followup_flags}" if followup_flags else ""
    capture_source = _capture_source(trace_client)
    return [
        {
            "cmd": f"engram axi context{project_flag} --budget 800{followup_suffix}",
            "reason": "Load compact workspace context",
        },
        {
            "cmd": f'engram axi recall "query"{project_flag} --limit 5{followup_suffix}',
            "reason": "Search long-tail memory",
        },
        {
            "cmd": f"engram axi observe --stdin --source {shlex.quote(capture_source)}",
            "reason": "Capture explicit user-approved notes",
        },
    ]


def _capture_source(trace_client: str | None) -> str:
    if trace_client and trace_client.strip():
        return trace_client.strip()
    return "axi"


def _followup_command_flags(
    *,
    server_url: str | None,
    timeout_seconds: float | None,
    trace_file: str | None,
    trace_client: str | None,
    followup_trace_origin: str | None,
) -> str:
    parts: list[str] = []
    has_trace = bool(trace_file and trace_client and followup_trace_origin)
    if has_trace and server_url:
        parts.extend(["--server-url", server_url])
    parts.extend(["--timeout", f"{FOLLOWUP_TIMEOUT_SECONDS:g}"])
    if has_trace:
        parts.extend(
            [
                "--trace-file",
                trace_file,
                "--trace-client",
                trace_client,
                "--trace-origin",
                followup_trace_origin,
            ]
        )
    return " ".join(shlex.quote(part) for part in parts)


def _compact_recall_results(
    results: Any,
    *,
    budget: int,
    full: bool,
) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        return []
    per_item_budget = max(80, int(budget / max(1, min(len(results), 5))))
    compact: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        result_type = item.get("resultType") or item.get("result_type") or item.get("type")
        nested = _nested_recall_payload(item, result_type=str(result_type or ""))
        text = (
            nested.get("summary")
            or nested.get("content")
            or item.get("summary")
            or item.get("content")
            or item.get("text")
            or item.get("snippet")
            or item.get("chunk_context")
            or ""
        )
        rendered, truncated, original_len = truncate_text(
            compact_whitespace(str(text)),
            budget_tokens=None if full else per_item_budget,
            minimum_chars=120,
        )
        compact.append(
            {
                "rank": item.get("rank") or index + 1,
                "type": result_type or "memory",
                "name": nested.get("name")
                or item.get("name")
                or item.get("entity_name")
                or item.get("title")
                or "",
                "score": item.get("score") or item.get("activation") or "",
                "text": rendered,
                "truncated": truncated,
                "original_chars": original_len,
            }
        )
    return compact


def _nested_recall_payload(item: dict[str, Any], *, result_type: str) -> dict[str, Any]:
    entity = item.get("entity")
    if isinstance(entity, dict):
        return {
            "name": entity.get("name"),
            "summary": entity.get("summary"),
        }
    episode = item.get("episode")
    if isinstance(episode, dict):
        return {
            "name": episode.get("source") or episode.get("id"),
            "content": episode.get("content"),
        }
    cue = item.get("cue") or item.get("episodeCue")
    if isinstance(cue, dict):
        return {
            "name": cue.get("episode_id") or cue.get("episodeId"),
            "content": cue.get("cue_text") or cue.get("cueText"),
        }
    if result_type == "packet" and isinstance(item.get("packet"), dict):
        packet = item["packet"]
        return {
            "name": packet.get("title"),
            "summary": packet.get("summary"),
        }
    return {}


def _compact_write_result(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    lifecycle = result.get("lifecycle") or {}
    payload = {
        "operation": operation,
        "status": result.get("status") or "ok",
        "episode_id": first_present(result, "episode_id", "episodeId"),
        "capture_status": lifecycle.get("capture_status"),
        "projection_mode": lifecycle.get("projection_mode"),
        "projection_status": lifecycle.get("projection_status"),
    }
    message = result.get("message")
    if message:
        payload["message"] = message
    return payload
