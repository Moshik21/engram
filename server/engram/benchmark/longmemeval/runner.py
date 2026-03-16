"""LongMemEval benchmark runner: load, ingest, query, evaluate, aggregate.

Zero LLM calls — uses embedding-based answer containment for evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from engram.benchmark.longmemeval.adapter import (
    AdapterStats,
    EngramLongMemEvalAdapter,
)
from engram.benchmark.longmemeval.dataset import (
    QUESTION_TYPES,
    LongMemEvalInstance,
    load_dataset,
)
from engram.benchmark.longmemeval.evaluator import (
    compute_containment_score,
    compute_retrieval_metrics,
    judge_by_containment,
)
from engram.config import ActivationConfig

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    """Complete result for one LongMemEval instance."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str
    correct: bool
    judge_raw: str
    evidence: list[str]
    evidence_scores: list[float]
    retrieved_session_ids: list[str]
    answer_session_ids: list[str]
    retrieval_metrics: dict[str, float]
    query_latency_ms: float
    ingest_sessions: int
    num_entities: int
    num_episodes: int
    containment_score: float = 0.0


@dataclass
class TypeMetrics:
    """Aggregated metrics for one question type."""

    question_type: str
    count: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    avg_recall_at_5: float
    avg_ndcg_at_5: float
    avg_containment: float = 0.0


@dataclass
class LongMemEvalResult:
    """Full benchmark result."""

    variant: str
    extraction_mode: str
    embedding_provider: str
    consolidation_used: bool
    total_instances: int
    total_correct: int
    overall_accuracy: float
    category_accuracy: float
    avg_containment: float
    type_metrics: list[TypeMetrics]
    instance_results: list[InstanceResult]
    adapter_stats: AdapterStats
    elapsed_seconds: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "variant": self.variant,
            "extraction_mode": self.extraction_mode,
            "embedding_provider": self.embedding_provider,
            "consolidation_used": self.consolidation_used,
            "assessment_method": "embedding_containment",
            "total_instances": self.total_instances,
            "total_correct": self.total_correct,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "category_accuracy": round(self.category_accuracy, 4),
            "avg_containment": round(self.avg_containment, 4),
            "type_metrics": [
                {
                    "question_type": tm.question_type,
                    "count": tm.count,
                    "correct": tm.correct,
                    "accuracy": round(tm.accuracy, 4),
                    "avg_latency_ms": round(tm.avg_latency_ms, 1),
                    "avg_recall_at_5": round(tm.avg_recall_at_5, 4),
                    "avg_ndcg_at_5": round(tm.avg_ndcg_at_5, 4),
                    "avg_containment": round(tm.avg_containment, 4),
                }
                for tm in self.type_metrics
            ],
            "adapter_stats": {
                "sessions_ingested": self.adapter_stats.sessions_ingested,
                "episodes_stored": self.adapter_stats.episodes_stored,
                "episodes_extracted": self.adapter_stats.episodes_extracted,
                "extraction_calls": self.adapter_stats.extraction_calls,
                "embedding_calls": self.adapter_stats.embedding_calls,
                "recall_calls": self.adapter_stats.recall_calls,
                "total_ingest_ms": round(self.adapter_stats.total_ingest_ms, 1),
                "total_query_ms": round(self.adapter_stats.total_query_ms, 1),
            },
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "instances": [
                {
                    "question_id": ir.question_id,
                    "question_type": ir.question_type,
                    "correct": ir.correct,
                    "hypothesis": ir.hypothesis,
                    "gold_answer": ir.gold_answer,
                    "containment_score": round(ir.containment_score, 4),
                    "retrieval_metrics": {k: round(v, 4) for k, v in ir.retrieval_metrics.items()},
                    "query_latency_ms": round(ir.query_latency_ms, 1),
                    "num_entities": ir.num_entities,
                    "num_episodes": ir.num_episodes,
                }
                for ir in self.instance_results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Results saved to %s", path)


async def run_longmemeval(
    dataset_path: str | Path,
    *,
    variant: str = "auto",
    extraction_mode: str = "narrow",
    embedding_provider: str = "local",
    consolidation: bool = False,
    containment_threshold: float = 0.72,
    top_k: int = 10,
    max_instances: int | None = None,
    n_per_type: int | None = None,
    question_types: list[str] | None = None,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    verbose: bool = False,
) -> LongMemEvalResult:
    """Run the full LongMemEval benchmark (zero LLM calls).

    Args:
        dataset_path: Path to the dataset JSON file.
        variant: Dataset variant (oracle/s/m/auto).
        extraction_mode: How to extract entities (none/narrow/full/auto).
        embedding_provider: Embedding provider (none/local/voyage/auto).
        consolidation: Whether to run consolidation after ingestion.
        containment_threshold: Cosine similarity threshold for correct.
        top_k: Number of results to retrieve per query.
        max_instances: Maximum total instances to process.
        n_per_type: If set, use stratified sampling with this many per type.
        question_types: Filter to specific question types.
        output_path: Path to save results JSON.
        checkpoint_path: Path for incremental checkpointing.
        verbose: Enable verbose logging.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    start = time.perf_counter()

    # Load dataset
    dataset = load_dataset(dataset_path, max_instances=max_instances, variant=variant)
    variant = dataset.variant

    # Apply filters
    if question_types:
        dataset = dataset.filter_types(question_types)
    if n_per_type:
        dataset = dataset.stratified_subset(n_per_type)
    elif max_instances and len(dataset.instances) > max_instances:
        dataset = dataset.subset(max_instances)

    logger.info(
        "Running LongMemEval (%s): %d instances, extraction=%s, embeddings=%s, "
        "assessment=embedding_containment (threshold=%.2f)",
        variant,
        len(dataset.instances),
        extraction_mode,
        embedding_provider,
        containment_threshold,
    )

    # Configure Engram
    cfg = ActivationConfig()

    # Create adapter
    adapter = EngramLongMemEvalAdapter(
        cfg=cfg,
        extraction_mode=extraction_mode,
        consolidation=consolidation,
        top_k=top_k,
    )

    # Load checkpoint if resuming
    completed_ids: set[str] = set()
    checkpointed_results: list[InstanceResult] = []
    if checkpoint_path:
        completed_ids, checkpointed_results = _load_checkpoint(checkpoint_path)

    # Process each instance
    instance_results: list[InstanceResult] = list(checkpointed_results)

    for i, instance in enumerate(dataset.instances):
        if instance.question_id in completed_ids:
            logger.info("Skipping %s (already checkpointed)", instance.question_id)
            continue

        logger.info(
            "[%d/%d] Processing %s (%s) -- %d sessions",
            i + 1,
            len(dataset.instances),
            instance.question_id,
            instance.question_type,
            instance.num_sessions,
        )

        try:
            result = await _process_instance(
                adapter=adapter,
                instance=instance,
                containment_threshold=containment_threshold,
            )
            instance_results.append(result)

            # Checkpoint incrementally
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, instance_results)

        except Exception:
            logger.error("Failed to process %s", instance.question_id, exc_info=True)
            # Record as incorrect
            instance_results.append(
                InstanceResult(
                    question_id=instance.question_id,
                    question_type=instance.question_type,
                    question=instance.question,
                    gold_answer=instance.answer,
                    hypothesis="[ERROR]",
                    correct=False,
                    judge_raw="error",
                    evidence=[],
                    evidence_scores=[],
                    retrieved_session_ids=[],
                    answer_session_ids=instance.answer_session_ids,
                    retrieval_metrics={},
                    query_latency_ms=0.0,
                    ingest_sessions=instance.num_sessions,
                    num_entities=0,
                    num_episodes=0,
                    containment_score=0.0,
                )
            )

    await adapter.close()
    elapsed = time.perf_counter() - start

    # Aggregate metrics
    type_metrics = _aggregate_type_metrics(instance_results)
    total_correct = sum(1 for r in instance_results if r.correct)
    overall_accuracy = total_correct / len(instance_results) if instance_results else 0.0

    # Official metric: unweighted average across categories
    non_abstention = [tm for tm in type_metrics if tm.question_type != "abstention"]
    category_accuracy = (
        sum(tm.accuracy for tm in non_abstention) / len(non_abstention) if non_abstention else 0.0
    )

    # Average containment score
    containment_scores = [r.containment_score for r in instance_results]
    avg_containment = (
        sum(containment_scores) / len(containment_scores) if containment_scores else 0.0
    )

    result = LongMemEvalResult(
        variant=variant,
        extraction_mode=extraction_mode,
        embedding_provider=embedding_provider,
        consolidation_used=consolidation,
        total_instances=len(instance_results),
        total_correct=total_correct,
        overall_accuracy=overall_accuracy,
        category_accuracy=category_accuracy,
        avg_containment=avg_containment,
        type_metrics=type_metrics,
        instance_results=instance_results,
        adapter_stats=adapter.stats,
        elapsed_seconds=elapsed,
    )

    if output_path:
        result.save(output_path)

    return result


async def _process_instance(
    adapter: EngramLongMemEvalAdapter,
    instance: LongMemEvalInstance,
    containment_threshold: float,
) -> InstanceResult:
    """Process a single LongMemEval instance: ingest, query, judge."""
    # Ingest all sessions
    await adapter.ingest_instance(instance)

    # Query
    query_result = await adapter.query_instance(instance)

    # Compute embedding-based containment score
    embed_fn = adapter.get_embed_fn()
    containment_score = 0.0
    if embed_fn and query_result.evidence:
        containment_score = await compute_containment_score(
            gold_answer=instance.answer,
            evidence_texts=query_result.evidence,
            embed_fn=embed_fn,
        )

    # Judge by containment
    verdict = judge_by_containment(
        question_id=instance.question_id,
        question_type=instance.question_type,
        containment_score=containment_score,
        is_abstention=instance.is_abstention,
        threshold=containment_threshold,
        hypothesis=query_result.hypothesis,
        gold_answer=instance.answer,
    )

    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        retrieved_session_ids=query_result.retrieved_session_ids,
        answer_session_ids=instance.answer_session_ids,
    )

    return InstanceResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        question=instance.question,
        gold_answer=instance.answer,
        hypothesis=query_result.hypothesis,
        correct=verdict.correct,
        judge_raw=verdict.judge_raw,
        evidence=query_result.evidence,
        evidence_scores=query_result.evidence_scores,
        retrieved_session_ids=query_result.retrieved_session_ids,
        answer_session_ids=instance.answer_session_ids,
        retrieval_metrics=retrieval_metrics,
        query_latency_ms=query_result.latency_ms,
        ingest_sessions=instance.num_sessions,
        num_entities=query_result.num_entities,
        num_episodes=query_result.num_episodes,
        containment_score=containment_score,
    )


def _aggregate_type_metrics(
    results: list[InstanceResult],
) -> list[TypeMetrics]:
    """Aggregate metrics by question type."""
    by_type: dict[str, list[InstanceResult]] = {}
    for r in results:
        # Group abstention questions separately
        key = "abstention" if r.question_id.endswith("_abs") else r.question_type
        by_type.setdefault(key, []).append(r)

    metrics = []
    for qtype in QUESTION_TYPES + ["abstention"]:
        type_results = by_type.get(qtype, [])
        if not type_results:
            continue

        correct = sum(1 for r in type_results if r.correct)
        latencies = [r.query_latency_ms for r in type_results]
        recall_5 = [r.retrieval_metrics.get("recall@5", 0.0) for r in type_results]
        ndcg_5 = [r.retrieval_metrics.get("ndcg@5", 0.0) for r in type_results]
        containments = [r.containment_score for r in type_results]

        metrics.append(
            TypeMetrics(
                question_type=qtype,
                count=len(type_results),
                correct=correct,
                accuracy=correct / len(type_results),
                avg_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
                avg_recall_at_5=(sum(recall_5) / len(recall_5) if recall_5 else 0.0),
                avg_ndcg_at_5=(sum(ndcg_5) / len(ndcg_5) if ndcg_5 else 0.0),
                avg_containment=(sum(containments) / len(containments) if containments else 0.0),
            )
        )

    return metrics


def _load_checkpoint(
    path: str | Path,
) -> tuple[set[str], list[InstanceResult]]:
    """Load completed instance IDs from a checkpoint file."""
    path = Path(path)
    if not path.exists():
        return set(), []

    try:
        with open(path) as f:
            data = json.load(f)
        completed: set[str] = set()
        results: list[InstanceResult] = []
        for entry in data.get("instances", []):
            qid = entry["question_id"]
            completed.add(qid)
            results.append(
                InstanceResult(
                    question_id=qid,
                    question_type=entry.get("question_type", ""),
                    question=entry.get("question", ""),
                    gold_answer=entry.get("gold_answer", ""),
                    hypothesis=entry.get("hypothesis", ""),
                    correct=entry.get("correct", False),
                    judge_raw=entry.get("judge_raw", ""),
                    evidence=entry.get("evidence", []),
                    evidence_scores=entry.get("evidence_scores", []),
                    retrieved_session_ids=entry.get("retrieved_session_ids", []),
                    answer_session_ids=entry.get("answer_session_ids", []),
                    retrieval_metrics=entry.get("retrieval_metrics", {}),
                    query_latency_ms=entry.get("query_latency_ms", 0.0),
                    ingest_sessions=entry.get("ingest_sessions", 0),
                    num_entities=entry.get("num_entities", 0),
                    num_episodes=entry.get("num_episodes", 0),
                    containment_score=entry.get("containment_score", 0.0),
                )
            )
        logger.info("Loaded checkpoint with %d completed instances", len(completed))
        return completed, results
    except Exception:
        logger.warning("Failed to load checkpoint from %s", path, exc_info=True)
        return set(), []


def _save_checkpoint(path: str | Path, results: list[InstanceResult]) -> None:
    """Save current progress to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "instances": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "hypothesis": r.hypothesis,
                "correct": r.correct,
                "judge_raw": r.judge_raw,
                "evidence": r.evidence[:3],
                "evidence_scores": r.evidence_scores[:10],
                "retrieved_session_ids": r.retrieved_session_ids,
                "answer_session_ids": r.answer_session_ids,
                "retrieval_metrics": r.retrieval_metrics,
                "query_latency_ms": r.query_latency_ms,
                "ingest_sessions": r.ingest_sessions,
                "num_entities": r.num_entities,
                "num_episodes": r.num_episodes,
                "containment_score": r.containment_score,
            }
            for r in results
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
