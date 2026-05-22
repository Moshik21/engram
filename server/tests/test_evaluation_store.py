from __future__ import annotations

import aiosqlite
import pytest

from engram.evaluation.store import (
    MEMORY_OPERATION_METRICS_RETENTION,
    RECALL_RUNTIME_SNAPSHOT_RETENTION,
    SQLiteEvaluationStore,
    StoredMemoryOperationMetricsSnapshot,
    StoredRecallEvalSample,
    StoredRecallRuntimeMetricsSnapshot,
    StoredSessionContinuitySample,
)


@pytest.mark.asyncio
async def test_evaluation_store_round_trips_recall_and_session_samples(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        await store.save_recall_sample(
            StoredRecallEvalSample(
                id="ers_one",
                group_id="default",
                recall_triggered=True,
                recall_helped=False,
                recall_needed=True,
                packets_surfaced=4,
                packets_used=1,
                false_recalls=2,
                stale_packets=3,
                corrected_packets=1,
                query="What did I decide?",
                timestamp=10.0,
            )
        )
        await store.save_session_sample(
            StoredSessionContinuitySample(
                id="esc_one",
                group_id="default",
                baseline_score=0.2,
                memory_score=0.8,
                open_loop_expected=True,
                open_loop_recovered=True,
                temporal_expected=True,
                temporal_correct=False,
                scenario="open loop follow-up",
                timestamp=11.0,
            )
        )

        recall_samples = await store.get_recall_samples("default")
        session_samples = await store.get_session_samples("default")

        assert len(recall_samples) == 1
        assert recall_samples[0].recall_triggered is True
        assert recall_samples[0].recall_helped is False
        assert recall_samples[0].recall_needed is True
        assert recall_samples[0].packets_surfaced == 4
        assert recall_samples[0].packets_used == 1
        assert recall_samples[0].false_recalls == 2
        assert recall_samples[0].stale_packets == 3
        assert recall_samples[0].corrected_packets == 1
        assert len(session_samples) == 1
        assert session_samples[0].baseline_score == 0.2
        assert session_samples[0].memory_score == 0.8
        assert session_samples[0].open_loop_expected is True
        assert session_samples[0].open_loop_recovered is True
        assert session_samples[0].temporal_expected is True
        assert session_samples[0].temporal_correct is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_store_prunes_recall_runtime_metrics_by_group(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        for index in range(RECALL_RUNTIME_SNAPSHOT_RETENTION + 5):
            await store.save_recall_metrics_snapshot(
                StoredRecallRuntimeMetricsSnapshot(
                    id=f"erm_default_{index}",
                    group_id="default",
                    metrics={"total_analyses": index},
                    source="test",
                    timestamp=float(index),
                )
            )
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                id="erm_other",
                group_id="other",
                metrics={"total_analyses": 99},
                source="test",
                timestamp=1.0,
            )
        )

        default_count = await (
            await store.db.execute(
                "SELECT COUNT(*) FROM evaluation_recall_runtime_metrics "
                "WHERE group_id = ?",
                ("default",),
            )
        ).fetchone()
        other_count = await (
            await store.db.execute(
                "SELECT COUNT(*) FROM evaluation_recall_runtime_metrics "
                "WHERE group_id = ?",
                ("other",),
            )
        ).fetchone()
        oldest_default = await (
            await store.db.execute(
                "SELECT MIN(timestamp) FROM evaluation_recall_runtime_metrics "
                "WHERE group_id = ?",
                ("default",),
            )
        ).fetchone()
        latest = await store.get_latest_recall_metrics_snapshot("default")

        assert default_count[0] == RECALL_RUNTIME_SNAPSHOT_RETENTION
        assert other_count[0] == 1
        assert oldest_default[0] == 5.0
        assert latest["total_analyses"] == RECALL_RUNTIME_SNAPSHOT_RETENTION + 4
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_store_does_not_close_borrowed_db_connection(tmp_path) -> None:
    db = await aiosqlite.connect(tmp_path / "engram.db")
    db.row_factory = aiosqlite.Row
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize(db=db)
    try:
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                group_id="default",
                metrics={"total_analyses": 1},
                source="test",
            )
        )
        await store.close()

        row = await (await db.execute("SELECT COUNT(*) FROM evaluation_recall_samples")).fetchone()
        assert row[0] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_evaluation_store_filters_by_group_and_limit(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        await store.save_recall_sample(
            StoredRecallEvalSample(
                id="ers_old",
                group_id="default",
                recall_triggered=True,
                recall_helped=True,
                timestamp=1.0,
            )
        )
        await store.save_recall_sample(
            StoredRecallEvalSample(
                id="ers_new",
                group_id="default",
                recall_triggered=True,
                recall_helped=False,
                timestamp=2.0,
            )
        )
        await store.save_recall_sample(
            StoredRecallEvalSample(
                id="ers_other",
                group_id="other",
                recall_triggered=True,
                recall_helped=False,
                timestamp=3.0,
            )
        )

        samples = await store.get_recall_samples("default", limit=1)

        assert len(samples) == 1
        assert samples[0].recall_helped is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_store_round_trips_recall_runtime_metrics(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                id="erm_old",
                group_id="default",
                metrics={"total_analyses": 1, "analyzer_latency_ms": {"p95": 12.0}},
                source="test",
                timestamp=1.0,
            )
        )
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                id="erm_new",
                group_id="default",
                metrics={
                    "total_analyses": 2,
                    "trigger_count": 1,
                    "analyzer_latency_ms": {"avg": 8.0, "p95": 16.0},
                    "surfaced_count": 3,
                },
                source="test",
                timestamp=2.0,
            )
        )
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                id="erm_other",
                group_id="other",
                metrics={"total_analyses": 9},
                source="test",
                timestamp=3.0,
            )
        )

        metrics = await store.get_latest_recall_metrics_snapshot("default")

        assert metrics["total_analyses"] == 2
        assert metrics["trigger_count"] == 1
        assert metrics["analyzer_latency_ms"]["p95"] == 16.0
        assert metrics["surfaced_count"] == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_store_round_trips_memory_operation_metrics(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        await store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                id="emo_old",
                group_id="default",
                metrics={"operation_count": 1, "duration_ms": {"p95": 12.0}},
                source="test",
                timestamp=1.0,
            )
        )
        await store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                id="emo_new",
                group_id="default",
                metrics={
                    "operation_count": 4,
                    "duration_ms": {"avg": 7.0, "p95": 18.0},
                    "timeout_count": 1,
                    "cache_hit_count": 3,
                    "cache_miss_count": 1,
                },
                source="test",
                timestamp=2.0,
            )
        )
        await store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                id="emo_other",
                group_id="other",
                metrics={"operation_count": 9},
                source="test",
                timestamp=3.0,
            )
        )

        metrics = await store.get_latest_memory_operation_metrics_snapshot("default")

        assert metrics["operation_count"] == 4
        assert metrics["duration_ms"]["p95"] == 18.0
        assert metrics["timeout_count"] == 1
        assert metrics["cache_hit_count"] == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_store_prunes_memory_operation_metrics_by_group(tmp_path) -> None:
    store = SQLiteEvaluationStore(str(tmp_path / "engram.db"))
    await store.initialize()
    try:
        for index in range(MEMORY_OPERATION_METRICS_RETENTION + 4):
            await store.save_memory_operation_metrics_snapshot(
                StoredMemoryOperationMetricsSnapshot(
                    id=f"emo_default_{index}",
                    group_id="default",
                    metrics={"operation_count": index},
                    source="test",
                    timestamp=float(index),
                )
            )
        await store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                id="emo_other",
                group_id="other",
                metrics={"operation_count": 99},
                source="test",
                timestamp=1.0,
            )
        )

        default_count = await (
            await store.db.execute(
                "SELECT COUNT(*) FROM evaluation_memory_operation_metrics "
                "WHERE group_id = ?",
                ("default",),
            )
        ).fetchone()
        other_count = await (
            await store.db.execute(
                "SELECT COUNT(*) FROM evaluation_memory_operation_metrics "
                "WHERE group_id = ?",
                ("other",),
            )
        ).fetchone()
        oldest_default = await (
            await store.db.execute(
                "SELECT MIN(timestamp) FROM evaluation_memory_operation_metrics "
                "WHERE group_id = ?",
                ("default",),
            )
        ).fetchone()
        latest = await store.get_latest_memory_operation_metrics_snapshot("default")

        assert default_count[0] == MEMORY_OPERATION_METRICS_RETENTION
        assert other_count[0] == 1
        assert oldest_default[0] == 4.0
        assert latest["operation_count"] == MEMORY_OPERATION_METRICS_RETENTION + 3
    finally:
        await store.close()
