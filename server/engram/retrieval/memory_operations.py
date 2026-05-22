"""Runtime memory operation metrics for value and latency reporting."""

from __future__ import annotations

import inspect
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass(frozen=True)
class MemoryOperationSample:
    """One measured memory operation."""

    operation: str
    source: str = "runtime"
    mode: str | None = None
    status: str = "ok"
    duration_ms: float = 0.0
    budget_ms: float | None = None
    budget_tokens: int | None = None
    skip_reason: str | None = None
    timeout: bool = False
    degraded: bool = False
    budget_miss: bool = False
    cache_hit: bool | None = None
    result_count: int = 0
    packet_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MemoryOperationMetricsCollector:
    """Rolling in-memory metrics for memory operation cost reporting."""

    def __init__(self, *, max_samples: int = 200) -> None:
        self._max_samples = max(10, int(max_samples))
        self._samples: dict[str, deque[MemoryOperationSample]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )

    def record(self, group_id: str, sample: MemoryOperationSample) -> None:
        """Record a measured operation."""
        self._samples[group_id].append(sample)

    def snapshot(self, group_id: str) -> dict[str, Any]:
        """Return aggregate metrics for the current rolling window."""
        return _aggregate_samples(list(self._samples.get(group_id) or []))


def memory_operation_sample_from_mapping(
    payload: Mapping[str, Any],
) -> MemoryOperationSample:
    """Build a sample from a loose dict-like caller payload."""
    return MemoryOperationSample(
        operation=str(_get(payload, "operation", default="memory")),
        source=str(_get(payload, "source", default="runtime")),
        mode=_optional_str(_get(payload, "mode", default=None)),
        status=str(_get(payload, "status", default="ok")),
        duration_ms=_float(_get(payload, "duration_ms", "durationMs")),
        budget_ms=_optional_float(_get(payload, "budget_ms", "budgetMs", default=None)),
        budget_tokens=_optional_int(
            _get(payload, "budget_tokens", "budgetTokens", default=None)
        ),
        skip_reason=_optional_str(_get(payload, "skip_reason", "skipReason", default=None)),
        timeout=bool(_get(payload, "timeout", default=False)),
        degraded=bool(_get(payload, "degraded", default=False)),
        budget_miss=bool(_get(payload, "budget_miss", "budgetMiss", default=False)),
        cache_hit=_optional_bool(_get(payload, "cache_hit", "cacheHit", default=None)),
        result_count=_int(_get(payload, "result_count", "resultCount")),
        packet_count=_int(_get(payload, "packet_count", "packetCount")),
        timestamp=_float(_get(payload, "timestamp", default=time.time())),
    )


async def record_manager_memory_operation(
    manager: Any,
    group_id: str,
    sample: MemoryOperationSample | Mapping[str, Any],
) -> None:
    """Record a memory operation through sync or async manager facades."""
    recorder = getattr(manager, "record_memory_operation", None)
    if not callable(recorder):
        return
    payload = sample if isinstance(sample, MemoryOperationSample) else dict(sample)
    result = cast(Callable[..., Any], recorder)(group_id, payload)
    if inspect.isawaitable(result):
        await result


def measured_memory_operation(
    *,
    operation: str,
    source: str,
    mode: str | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> tuple[float, Callable[..., MemoryOperationSample]]:
    """Return a start timestamp plus a sample builder for simple call timers."""
    started = perf_counter()

    def finish(
        *,
        status: str = "ok",
        timeout: bool = False,
        degraded: bool = False,
        budget_miss: bool = False,
        skip_reason: str | None = None,
        budget_ms: float | None = None,
        budget_tokens: int | None = None,
        cache_hit: bool | None = None,
        result_count: int = 0,
        packet_count: int = 0,
    ) -> MemoryOperationSample:
        return MemoryOperationSample(
            operation=operation,
            source=source,
            mode=mode,
            status=status,
            duration_ms=round((perf_counter() - started) * 1000, 4),
            budget_ms=budget_ms,
            budget_tokens=budget_tokens,
            skip_reason=skip_reason,
            timeout=timeout,
            degraded=degraded,
            budget_miss=budget_miss,
            cache_hit=cache_hit,
            result_count=result_count,
            packet_count=packet_count,
        )

    return started, finish


def _aggregate_samples(samples: list[MemoryOperationSample]) -> dict[str, Any]:
    if not samples:
        return {}
    return {
        **_operation_summary(samples),
        "by_mode": {
            mode: _operation_summary(mode_samples)
            for mode, mode_samples in _grouped_samples(samples).items()
        },
    }


def _grouped_samples(
    samples: list[MemoryOperationSample],
) -> dict[str, list[MemoryOperationSample]]:
    grouped: dict[str, list[MemoryOperationSample]] = defaultdict(list)
    for sample in samples:
        key = sample.mode or sample.source or sample.operation
        grouped[key].append(sample)
    return dict(grouped)


def _operation_summary(samples: list[MemoryOperationSample]) -> dict[str, Any]:
    durations = [max(0.0, sample.duration_ms) for sample in samples]
    budget_ms_values = [
        max(0.0, sample.budget_ms)
        for sample in samples
        if sample.budget_ms is not None and sample.budget_ms > 0
    ]
    budget_token_values = [
        max(0, sample.budget_tokens)
        for sample in samples
        if sample.budget_tokens is not None and sample.budget_tokens > 0
    ]
    cache_hits = [sample.cache_hit for sample in samples if sample.cache_hit is not None]
    status_counts = _status_counts(samples)
    timeout_count = sum(
        1 for sample in samples if sample.timeout or sample.status == "timeout"
    )
    degraded_count = sum(
        1 for sample in samples if sample.degraded or sample.status == "degraded"
    )
    return {
        "operation_count": len(samples),
        "duration_ms": _latency_summary(durations),
        "budget_ms": _latency_summary(budget_ms_values),
        "avg_budget_tokens": _average_int(budget_token_values),
        "completed_count": status_counts.get("ok", 0) + status_counts.get("completed", 0),
        "skipped_count": status_counts.get("skipped", 0),
        "timeout_count": timeout_count,
        "degraded_count": degraded_count,
        "error_count": status_counts.get("error", 0),
        "budget_miss_count": sum(1 for sample in samples if sample.budget_miss),
        "cache_hit_count": sum(1 for hit in cache_hits if hit),
        "cache_miss_count": sum(1 for hit in cache_hits if not hit),
        "status_counts": status_counts,
        "skip_reason_counts": _skip_reason_counts(samples),
        "operation_counts": _field_counts(samples, "operation"),
        "source_counts": _field_counts(samples, "source"),
        "result_count": sum(max(0, sample.result_count) for sample in samples),
        "packet_count": sum(max(0, sample.packet_count) for sample in samples),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "avg": round(sum(ordered) / len(ordered), 4),
        "p95": round(ordered[p95_index], 4),
    }


def _status_counts(samples: list[MemoryOperationSample]) -> dict[str, int]:
    return _field_counts(samples, "status")


def _skip_reason_counts(samples: list[MemoryOperationSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        if not sample.skip_reason:
            continue
        counts[sample.skip_reason] = counts.get(sample.skip_reason, 0) + 1
    return counts


def _field_counts(samples: list[MemoryOperationSample], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        value = str(getattr(sample, field_name) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _get(source: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return default


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _average_int(values: list[int]) -> int:
    if not values:
        return 0
    return round(sum(values) / len(values))
