"""LoCoMo benchmark runner: load → ingest → probe → evaluate → aggregate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram.benchmark.locomo.adapter import (
    conversation_to_episodes,
    load_locomo_dataset,
    probes_to_queries,
)
from engram.benchmark.locomo.answer_composer import compose_answer
from engram.benchmark.locomo.metrics import exact_match, token_f1
from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single probe evaluation."""

    probe_question: str
    ground_truth: str
    composed_answer: str
    exact_match_score: float
    f1_score: float
    category: str = ""


@dataclass
class ConversationResult:
    """Aggregate results for one conversation."""

    conversation_id: str
    num_turns: int
    num_probes: int
    probe_results: list[ProbeResult]
    avg_em: float
    avg_f1: float


@dataclass
class LoCoMoResult:
    """Full LoCoMo benchmark result."""

    total_conversations: int
    total_probes: int
    conversation_results: list[ConversationResult]
    overall_em: float
    overall_f1: float
    elapsed_seconds: float = 0.0
    category_scores: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


async def run_locomo(
    dataset_path: str | Path,
    graph_store,
    activation_store,
    search_index,
    cfg: ActivationConfig,
    group_id: str = "default",
    max_conversations: int | None = None,
    limit: int = 3,
) -> LoCoMoResult:
    """Run the full LoCoMo benchmark pipeline.

    1. Load dataset
    2. For each conversation: ingest turns as episodes
    3. Run probes via recall pipeline
    4. Compose answers from retrieved entity summaries
    5. Evaluate with EM and F1
    6. Aggregate results
    """
    start = time.perf_counter()

    # Load dataset
    conversations = load_locomo_dataset(
        dataset_path, max_conversations=max_conversations
    )

    conversation_results = []
    all_em_scores: list[float] = []
    all_f1_scores: list[float] = []
    category_scores: dict[str, list[tuple[float, float]]] = {}

    for conv in conversations:
        # Ingest turns as episodes
        episodes = conversation_to_episodes(conv, group_id=group_id)
        for ep in episodes:
            await graph_store.create_episode(ep)
            await search_index.index_episode(ep)

        # Run probes
        probes = probes_to_queries(conv.probes)
        probe_results = []

        for question, answer, category in probes:
            # Retrieve relevant entities
            results = await retrieve(
                query=question,
                group_id=group_id,
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=search_index,
                cfg=cfg,
                limit=limit,
                enable_routing=False,
            )

            # Get entity summaries
            summaries = []
            for r in results:
                entity = await graph_store.get_entity(
                    r.node_id, group_id
                )
                if entity and entity.summary:
                    summaries.append(entity.summary)

            composed = compose_answer(summaries)
            em = exact_match(composed, answer)
            f1 = token_f1(composed, answer)

            probe_results.append(ProbeResult(
                probe_question=question,
                ground_truth=answer,
                composed_answer=composed,
                exact_match_score=em,
                f1_score=f1,
                category=category,
            ))

            all_em_scores.append(em)
            all_f1_scores.append(f1)

            if category:
                category_scores.setdefault(category, []).append(
                    (em, f1)
                )

        # Conversation-level aggregation
        conv_em = (
            sum(p.exact_match_score for p in probe_results) / len(probe_results)
            if probe_results else 0.0
        )
        conv_f1 = (
            sum(p.f1_score for p in probe_results) / len(probe_results)
            if probe_results else 0.0
        )

        conversation_results.append(ConversationResult(
            conversation_id=conv.conversation_id,
            num_turns=len(conv.turns),
            num_probes=len(probes),
            probe_results=probe_results,
            avg_em=conv_em,
            avg_f1=conv_f1,
        ))

    # Overall aggregation
    overall_em = (
        sum(all_em_scores) / len(all_em_scores)
        if all_em_scores else 0.0
    )
    overall_f1 = (
        sum(all_f1_scores) / len(all_f1_scores)
        if all_f1_scores else 0.0
    )

    # Category-level aggregation
    cat_agg = {}
    for cat, scores in category_scores.items():
        em_vals = [s[0] for s in scores]
        f1_vals = [s[1] for s in scores]
        cat_agg[cat] = {
            "em": sum(em_vals) / len(em_vals) if em_vals else 0.0,
            "f1": sum(f1_vals) / len(f1_vals) if f1_vals else 0.0,
            "count": len(scores),
        }

    elapsed = time.perf_counter() - start

    return LoCoMoResult(
        total_conversations=len(conversations),
        total_probes=len(all_em_scores),
        conversation_results=conversation_results,
        overall_em=overall_em,
        overall_f1=overall_f1,
        elapsed_seconds=elapsed,
        category_scores=cat_agg,
    )
