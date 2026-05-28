from __future__ import annotations

from engram.retrieval.memory_operations import (
    MemoryOperationMetricsCollector,
    MemoryOperationSample,
    memory_operation_sample_from_mapping,
)


def test_memory_operation_metrics_collector_aggregates_costs_by_mode() -> None:
    collector = MemoryOperationMetricsCollector(max_samples=10)

    collector.record(
        "default",
        MemoryOperationSample(
            operation="context",
            source="axi_context",
            mode="axi_context",
            duration_ms=20.0,
            budget_ms=250.0,
            budget_tokens=300,
            result_count=3,
        ),
    )
    collector.record(
        "default",
        MemoryOperationSample(
            operation="recall",
            source="mcp_recall",
            mode="mcp_recall",
            duration_ms=40.0,
            budget_ms=30.0,
            budget_tokens=600,
            timeout=True,
            budget_miss=True,
            cache_hit=False,
            result_count=2,
            packet_count=1,
        ),
    )
    collector.record(
        "default",
        MemoryOperationSample(
            operation="auto_recall_gate",
            source="auto_recall",
            mode="auto_recall",
            status="skipped",
            duration_ms=1.0,
            budget_ms=750.0,
            budget_tokens=300,
            skip_reason="skipped_low_signal",
        ),
    )
    collector.record(
        "other",
        MemoryOperationSample(operation="recall", source="api_recall", duration_ms=99.0),
    )

    metrics = collector.snapshot("default")

    assert metrics["operation_count"] == 3
    assert metrics["duration_ms"] == {"avg": 20.3333, "p95": 40.0}
    assert metrics["budget_ms"] == {"avg": 343.3333, "p95": 750.0}
    assert metrics["avg_budget_tokens"] == 400
    assert metrics["completed_count"] == 2
    assert metrics["skipped_count"] == 1
    assert metrics["error_count"] == 0
    assert metrics["skip_reason_counts"] == {"skipped_low_signal": 1}
    assert metrics["timeout_count"] == 1
    assert metrics["budget_miss_count"] == 1
    assert metrics["cache_hit_count"] == 0
    assert metrics["cache_miss_count"] == 1
    assert metrics["operation_counts"] == {
        "context": 1,
        "recall": 1,
        "auto_recall_gate": 1,
    }
    assert metrics["source_counts"] == {
        "axi_context": 1,
        "mcp_recall": 1,
        "auto_recall": 1,
    }
    assert metrics["recent_problem_samples"] == [
        {
            "operation": "recall",
            "source": "mcp_recall",
            "mode": "mcp_recall",
            "status": "ok",
            "duration_ms": 40.0,
            "age_seconds": metrics["recent_problem_samples"][0]["age_seconds"],
            "budget_ms": 30.0,
            "budget_tokens": 600,
            "timeout": True,
            "budget_miss": True,
            "cache_hit": False,
            "result_count": 2,
            "packet_count": 1,
        }
    ]
    assert metrics["result_count"] == 5
    assert metrics["packet_count"] == 1
    assert metrics["by_mode"]["axi_context"]["operation_count"] == 1
    assert metrics["by_mode"]["mcp_recall"]["timeout_count"] == 1
    assert metrics["by_mode"]["auto_recall"]["skipped_count"] == 1
    assert collector.snapshot("missing") == {}


def test_memory_operation_sample_from_mapping_accepts_public_keys() -> None:
    sample = memory_operation_sample_from_mapping(
        {
            "operation": "recall",
            "source": "api_recall",
            "mode": "deep",
            "status": "error",
            "durationMs": 12.5,
            "budgetMs": 900,
            "budgetTokens": 500,
            "skipReason": "error",
            "budgetMiss": True,
            "cacheHit": True,
            "resultCount": 4,
            "packetCount": 2,
        }
    )

    assert sample.operation == "recall"
    assert sample.source == "api_recall"
    assert sample.mode == "deep"
    assert sample.status == "error"
    assert sample.duration_ms == 12.5
    assert sample.budget_ms == 900
    assert sample.budget_tokens == 500
    assert sample.skip_reason == "error"
    assert sample.budget_miss is True
    assert sample.cache_hit is True
    assert sample.result_count == 4
    assert sample.packet_count == 2
