"""Planner-driven multi-intent retrieval support."""

from __future__ import annotations

import asyncio

from engram.config import ActivationConfig
from engram.models.recall import RecallIntent, RecallPlan, RecallTrace


def build_recall_plan(
    query: str,
    cfg: ActivationConfig,
    *,
    conv_context=None,
    mode: str = "explicit_recall",
    memory_need=None,
) -> RecallPlan:
    """Build a bounded recall plan from the active query and session context."""
    intents = [
        RecallIntent(
            intent_type="direct",
            query_text=query.strip(),
            weight=1.0,
            candidate_budget=max(5, cfg.retrieval_top_k),
        )
    ]

    if cfg.conv_multi_query_enabled and conv_context is not None:
        turn_count = conv_context._turn_count
        if turn_count < 3:
            w_topic, w_entity = 0.25, 0.15
        else:
            w_topic, w_entity = 0.35, 0.30

        sub_limit = min(
            cfg.recall_planner_subquery_limit,
            max(5, cfg.retrieval_top_k // 2),
        )
        recent_turns = conv_context.get_recent_turns(cfg.conv_multi_query_turns)
        topic_query = " ".join(recent_turns).strip()
        entity_query = " ".join(
            entry.name for entry in conv_context.get_top_entities(cfg.conv_multi_query_top_entities)
        ).strip()

        intents = _append_unique_intent(
            intents,
            RecallIntent(
                intent_type="topic",
                query_text=topic_query,
                weight=w_topic,
                candidate_budget=sub_limit,
            ),
            query,
        )
        intents = _append_unique_intent(
            intents,
            RecallIntent(
                intent_type="session_entity",
                query_text=entity_query,
                weight=w_entity,
                candidate_budget=sub_limit,
            ),
            query,
        )

    return RecallPlan(
        query=query,
        mode=mode,
        intents=intents[: cfg.recall_planner_max_intents],
        seed_entity_ids=_dedupe_seed_entities(getattr(memory_need, "detected_entities", None)),
    )


async def execute_recall_plan(
    plan: RecallPlan,
    *,
    group_id: str,
    search_index,
    base_candidates: list[tuple[str, float]] | None = None,
) -> RecallTrace:
    """Execute planner intents and merge support additively."""
    trace = RecallTrace(plan=plan)
    merged_scores: dict[str, float] = {}

    direct_intent = next(
        (intent for intent in plan.intents if intent.intent_type == "direct"),
        None,
    )
    if direct_intent is not None and base_candidates is not None:
        _merge_results(
            merged_scores,
            trace,
            direct_intent,
            base_candidates,
        )
    elif direct_intent is not None and direct_intent.query_text:
        direct_results = await search_index.search(
            query=direct_intent.query_text,
            group_id=group_id,
            limit=direct_intent.candidate_budget,
        )
        _merge_results(merged_scores, trace, direct_intent, direct_results or [])

    sub_intents = [
        intent
        for intent in plan.intents
        if intent.intent_type != "direct" and intent.query_text
    ]
    if sub_intents:
        tasks = [
            search_index.search(
                query=intent.query_text,
                group_id=group_id,
                limit=intent.candidate_budget,
            )
            for intent in sub_intents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for intent, result in zip(sub_intents, results):
            if isinstance(result, list) and result:
                _merge_results(merged_scores, trace, intent, result)

    _inject_seed_entities(merged_scores, trace, plan)

    trace.merged_candidates = sorted(
        merged_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return trace


def merge_support(existing: float, contribution: float) -> float:
    """Bounded additive merge using noisy-or semantics."""
    existing = max(0.0, min(existing, 1.0))
    contribution = max(0.0, min(contribution, 1.0))
    return 1.0 - ((1.0 - existing) * (1.0 - contribution))


def _append_unique_intent(
    intents: list[RecallIntent],
    candidate: RecallIntent,
    direct_query: str,
) -> list[RecallIntent]:
    query_text = candidate.query_text.strip()
    if not query_text:
        return intents
    if query_text.lower() == direct_query.strip().lower():
        return intents
    existing_queries = {intent.query_text.lower() for intent in intents}
    if query_text.lower() in existing_queries:
        return intents
    return intents + [candidate]


def _merge_results(
    merged_scores: dict[str, float],
    trace: RecallTrace,
    intent: RecallIntent,
    results: list[tuple[str, float]],
) -> None:
    for entity_id, score in results:
        contribution = max(0.0, min(score * intent.weight, 1.0))
        merged_scores[entity_id] = merge_support(
            merged_scores.get(entity_id, 0.0),
            contribution,
        )
        trace.support_scores[entity_id] = merge_support(
            trace.support_scores.get(entity_id, 0.0),
            contribution,
        )
        if entity_id not in trace.intent_types:
            trace.intent_types[entity_id] = []
        if intent.intent_type not in trace.intent_types[entity_id]:
            trace.intent_types[entity_id].append(intent.intent_type)
        trace.support_details.setdefault(entity_id, []).append(
            {
                "intent_type": intent.intent_type,
                "query": intent.query_text,
                "weight": round(intent.weight, 4),
                "contribution": round(contribution, 4),
            }
        )


def _inject_seed_entities(
    merged_scores: dict[str, float],
    trace: RecallTrace,
    plan: RecallPlan,
) -> None:
    for entity_id in plan.seed_entity_ids:
        if entity_id in merged_scores:
            continue
        contribution = 0.14
        merged_scores[entity_id] = contribution
        trace.support_scores[entity_id] = contribution
        trace.intent_types.setdefault(entity_id, []).append("seed_entity")
        trace.support_details.setdefault(entity_id, []).append(
            {
                "intent_type": "seed_entity",
                "query": plan.query,
                "weight": 1.0,
                "contribution": round(contribution, 4),
            }
        )


def _dedupe_seed_entities(entity_ids: list[str] | None) -> list[str]:
    if not entity_ids:
        return []
    seen: set[str] = set()
    deduped: list[str] = []
    for entity_id in entity_ids:
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        deduped.append(entity_id)
    return deduped
