"""LoCoMo benchmark runner: load → ingest → probe → evaluate → aggregate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from engram.benchmark.locomo.adapter import (
    LOCOMO_BENCHMARK_GROUP_ID,
    category_label,
    conversation_to_session_contents,
    load_locomo_dataset,
)
from engram.benchmark.locomo.answer_composer import compose_answer
from engram.benchmark.locomo.metrics import exact_match, token_f1

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
    category_label: str = ""
    llm_correct: bool | None = None


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
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_llm_accuracy: float | None = None
    reader_mode: str = "none"
    judge_mode: str = "f1"
    reader_model: str = ""
    use_graph: bool = False


def _evidence_from_results(results: list, k: int) -> list[str]:
    """Flatten recall results into evidence strings (episode text + entity summaries)."""
    ev: list[str] = []
    for r in results[:k]:
        if not isinstance(r, dict):
            continue
        ep = r.get("episode") or {}
        ent = r.get("entity") or {}
        if ep.get("content"):
            ev.append(str(ep["content"]))
        elif ent.get("name"):
            ev.append(f"{ent['name']}: {ent.get('summary', '')}".strip())
    return ev


async def run_locomo(
    dataset_path: str | Path,
    *,
    group_id: str = LOCOMO_BENCHMARK_GROUP_ID,
    max_conversations: int | None = None,
    max_questions: int | None = None,
    limit: int = 10,
    reader: str = "none",
    judge: str = "f1",
    reader_model: str = "claude-sonnet-4-6",
    use_graph: bool = False,
    extraction_mode: str = "narrow",
) -> LoCoMoResult:
    """Run the LoCoMo benchmark through Engram's real ingest + recall pipeline.

    For each conversation: ingest each dated session as an episode (store_episode +
    project_episode, so the graph is populated), then for each question recall
    evidence and answer it (deterministic compose, or an LLM reader), scored by
    token-F1/EM and optionally an LLM judge. Stratified by LoCoMo category.
    """
    from engram.benchmark.longmemeval.adapter import (
        EngramLongMemEvalAdapter,
        _parse_session_date,
    )
    from engram.config import EngramConfig

    start = time.perf_counter()
    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)

    cfg_full = EngramConfig()
    adapter = EngramLongMemEvalAdapter(
        cfg=cfg_full.activation,
        extraction_mode=extraction_mode,
        embedding_provider="local",
        reranker_provider="local",
        use_graph=use_graph,
        shared_group=True,
    )
    await adapter._ensure_initialized()

    reader_judge = None
    if reader == "llm" or judge == "llm":
        if reader_model.startswith("claude"):
            from engram.benchmark.longmemeval.reader import LLMReaderJudge

            reader_judge = LLMReaderJudge(model=reader_model)
        else:
            # Fully-local reader/judge: any non-claude model name routes to Ollama
            # (base URL from the same env the fully-local config uses). Lets us grade
            # answer accuracy with zero external key / credits.
            import os

            from engram.benchmark.longmemeval.reader import OllamaReaderJudge

            reader_judge = OllamaReaderJudge(
                model=reader_model,
                base_url=os.environ.get(
                    "ENGRAM_ACTIVATION__OLLAMA_BASE_URL", "http://localhost:11434"
                ),
            )

    conversation_results = []
    all_em_scores: list[float] = []
    all_f1_scores: list[float] = []
    all_llm: list[float] = []
    category_scores: dict[str, list[tuple[float, float]]] = {}

    for conv in conversations:
        # Each conversation is an isolated memory; one group per conversation.
        await adapter._setup_manager(f"locomo_{conv.conversation_id}")
        # The lite store persists across runs — clear any prior ingest of this
        # conversation so retrieval isn't polluted by stale duplicate sessions.
        await adapter.cleanup_group(f"locomo_{conv.conversation_id}")
        mgr = adapter._manager
        gid = adapter._current_group_id

        # Ingest each dated session as an episode (with extraction).
        for sid, date, content in conversation_to_session_contents(conv):
            ep_id = await mgr.store_episode(
                content,
                group_id=gid,
                source=f"locomo:{conv.conversation_id}:{sid}",
                session_id=sid,
                conversation_date=_parse_session_date(date),
            )
            try:
                await mgr.project_episode(ep_id, group_id=gid)
            except Exception as e:
                logger.warning("projection failed for %s/%s: %s", conv.conversation_id, sid, e)

        probes = conv.probes[:max_questions] if max_questions else conv.probes
        probe_results = []

        for probe in probes:
            if not probe.question or not probe.answer:
                continue
            results = await mgr.recall(
                probe.question, group_id=gid, limit=limit, record_access=False,
            )
            evidence = _evidence_from_results(results, limit)

            if reader == "llm" and reader_judge is not None:
                answer = await reader_judge.read(probe.question, evidence)
            else:
                answer = compose_answer(evidence)

            em = exact_match(answer, probe.answer)
            f1 = token_f1(answer, probe.answer)
            label = category_label(probe.category)

            llm_correct = None
            if judge == "llm" and reader_judge is not None:
                llm_correct, _ = await reader_judge.judge(
                    question=probe.question,
                    hypothesis=answer,
                    gold_answer=probe.answer,
                    is_abstention=(label == "adversarial"),
                )
                all_llm.append(1.0 if llm_correct else 0.0)

            probe_results.append(
                ProbeResult(
                    probe_question=probe.question,
                    ground_truth=probe.answer,
                    composed_answer=answer,
                    exact_match_score=em,
                    f1_score=f1,
                    category=probe.category,
                    category_label=label,
                    llm_correct=llm_correct,
                )
            )
            all_em_scores.append(em)
            all_f1_scores.append(f1)
            # Per-category primary metric: LLM-judge accuracy when judged, else EM.
            if llm_correct is not None:
                primary = 1.0 if llm_correct else 0.0
            else:
                primary = em
            category_scores.setdefault(label, []).append((primary, f1))

        n = len(probe_results)
        conv_em = sum(p.exact_match_score for p in probe_results) / n if n else 0.0
        conv_f1 = sum(p.f1_score for p in probe_results) / n if n else 0.0
        conversation_results.append(
            ConversationResult(
                conversation_id=conv.conversation_id,
                num_turns=len(conv.turns),
                num_probes=len(probe_results),
                probe_results=probe_results,
                avg_em=conv_em,
                avg_f1=conv_f1,
            )
        )

    await adapter.close()
    if reader_judge is not None:
        await reader_judge.close()

    overall_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0.0
    overall_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
    overall_llm = sum(all_llm) / len(all_llm) if all_llm else None

    # Category aggregation: primary metric is LLM-accuracy when judged, else F1.
    cat_agg = {}
    for cat, scores in category_scores.items():
        prim = [s[0] for s in scores]
        f1_vals = [s[1] for s in scores]
        cat_agg[cat] = {
            "accuracy" if all_llm else "em": sum(prim) / len(prim) if prim else 0.0,
            "f1": sum(f1_vals) / len(f1_vals) if f1_vals else 0.0,
            "count": len(scores),
        }

    return LoCoMoResult(
        total_conversations=len(conversations),
        total_probes=len(all_em_scores),
        conversation_results=conversation_results,
        overall_em=overall_em,
        overall_f1=overall_f1,
        elapsed_seconds=time.perf_counter() - start,
        category_scores=cat_agg,
        overall_llm_accuracy=overall_llm,
        reader_mode=reader,
        judge_mode=judge,
        reader_model=reader_model if (reader == "llm" or judge == "llm") else "",
        use_graph=use_graph,
    )
