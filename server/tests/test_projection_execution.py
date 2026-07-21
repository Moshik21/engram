"""Tests for projection execution path helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.evidence import CommitDecision, EvidenceBundle, EvidenceCandidate
from engram.extraction.models import (
    ApplyOutcome,
    ClaimCandidate,
    EntityCandidate,
    ProjectionBundle,
    ProjectionPlan,
)
from engram.ingestion.projection_execution import (
    EvidenceProjectionExecutor,
    LegacyProjectionExecutor,
    ProjectionError,
    ProjectionLifecycleResult,
)
from engram.ingestion.projection_service import EpisodeProjectionService
from engram.models.consolidation import RelationshipApplyResult
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus


def _plan() -> ProjectionPlan:
    selected_text = "Python is used for AI."
    return ProjectionPlan(
        episode_id="ep_test",
        strategy="full",
        spans=[],
        selected_text=selected_text,
        selected_chars=len(selected_text),
        total_chars=len(selected_text),
    )


def _episode() -> Episode:
    return Episode(
        id="ep_test",
        content="Python is used for AI.",
        source="test",
        status=EpisodeStatus.QUEUED,
        group_id="default",
    )


def test_projection_lifecycle_result_event_payload_preserves_legacy_keys():
    result = ProjectionLifecycleResult(
        episode_id="ep_test",
        group_id="default",
        outcome="projected",
        episode_status=EpisodeStatus.COMPLETED,
        projection_state=EpisodeProjectionState.PROJECTED,
        reason="projected",
        entity_count=2,
        relationship_count=1,
        duration_ms=25,
        used_evidence_materializer=True,
        plan_strategy="full",
    )

    assert result.to_event_payload() == {
        "episodeId": "ep_test",
        "status": "completed",
        "outcome": "projected",
        "lifecycleStage": "project",
        "projectionState": "projected",
        "entity_count": 2,
        "relationship_count": 1,
        "reason": "projected",
        "duration_ms": 25,
        "planStrategy": "full",
        "usedEvidenceMaterializer": True,
    }


def test_projection_service_records_write_through_storage_deltas():
    calls = []
    service = EpisodeProjectionService(
        graph_store=AsyncMock(),
        cfg=ActivationConfig(),
        projection_planner=MagicMock(),
        evidence_projection_executor=AsyncMock(),
        legacy_projection_executor=AsyncMock(),
        content_hashes=set(),
        content_hashes_inflight=set(),
        update_episode_status=AsyncMock(),
        sync_projection_state=AsyncMock(),
        get_episode_cue=AsyncMock(),
        publish_event=MagicMock(),
        should_use_evidence_pipeline=MagicMock(return_value=False),
        run_surprise_detection=AsyncMock(),
        run_prospective_memory=AsyncMock(),
        publish_projection_graph_changes=AsyncMock(),
        index_projected_bundle=AsyncMock(),
        store_emotional_encoding_context=AsyncMock(),
        invalidate_briefing_cache=MagicMock(),
        record_storage_counts=lambda group_id, **counts: calls.append((group_id, counts)),
    )

    service._record_projection_storage_counts(
        "brain",
        ApplyOutcome(
            new_entity_names=["Engram", "AXI"],
            relationship_results=[
                RelationshipApplyResult(created=True),
                RelationshipApplyResult(created=False),
                RelationshipApplyResult(created=True),
            ],
        ),
    )

    assert calls == [
        ("brain", {"entities": 2, "relationships": 2}),
    ]


@pytest.mark.asyncio
async def test_legacy_projection_executor_applies_entities_and_relationships():
    plan = _plan()
    episode = _episode()
    bundle = ProjectionBundle(
        episode_id="ep_test",
        plan=plan,
        entities=[
            EntityCandidate(
                name="Python",
                entity_type="Technology",
                summary="A programming language",
            ),
        ],
        claims=[
            ClaimCandidate(
                subject_text="Python",
                predicate="USED_FOR",
                object_text="AI",
            ),
        ],
    )
    projector = AsyncMock()
    projector.project = AsyncMock(return_value=bundle)
    apply_engine = AsyncMock()
    apply_engine.apply_entities = AsyncMock(
        return_value=ApplyOutcome(entity_map={"Python": "ent_python"}),
    )
    apply_engine.apply_relationships = AsyncMock(return_value=[])
    update_episode_status = AsyncMock()
    apply_bootstrap_part_of_edges = AsyncMock()
    executor = LegacyProjectionExecutor(
        projector=projector,
        apply_engine=apply_engine,
        update_episode_status=update_episode_status,
        apply_bootstrap_part_of_edges=apply_bootstrap_part_of_edges,
    )

    outcome = await executor.execute(
        episode=episode,
        plan=plan,
        group_id="default",
    )

    assert outcome.bundle is bundle
    assert outcome.apply_outcome.entity_map == {"Python": "ent_python"}
    assert outcome.entity_map == {"Python": "ent_python"}
    assert outcome.used_evidence_materializer is False
    update_episode_status.assert_any_await(
        "ep_test",
        EpisodeStatus.RESOLVING,
        group_id="default",
    )
    update_episode_status.assert_any_await(
        "ep_test",
        EpisodeStatus.WRITING,
        group_id="default",
    )
    apply_engine.apply_entities.assert_awaited_once_with(
        bundle.entities,
        episode,
        "default",
        recall_content=plan.selected_text,
    )
    apply_bootstrap_part_of_edges.assert_awaited_once_with(
        episode,
        {"Python": "ent_python"},
        "default",
    )
    apply_engine.apply_relationships.assert_awaited_once_with(
        bundle.claims,
        entity_map={"Python": "ent_python"},
        meta_entity_names=set(),
        group_id="default",
        source_episode="ep_test",
        conversation_date=None,
    )


@pytest.mark.asyncio
async def test_legacy_projection_executor_raises_projection_error_for_retryable_bundle():
    plan = _plan()
    bundle = ProjectionBundle(
        episode_id="ep_test",
        plan=plan,
        entities=[],
        claims=[],
        extractor_status="api_error",
        extractor_error="API down",
        retryable=True,
    )
    projector = AsyncMock()
    projector.project = AsyncMock(return_value=bundle)
    apply_engine = AsyncMock()
    update_episode_status = AsyncMock()
    apply_bootstrap_part_of_edges = AsyncMock()
    executor = LegacyProjectionExecutor(
        projector=projector,
        apply_engine=apply_engine,
        update_episode_status=update_episode_status,
        apply_bootstrap_part_of_edges=apply_bootstrap_part_of_edges,
    )

    with pytest.raises(ProjectionError) as exc:
        await executor.execute(
            episode=_episode(),
            plan=plan,
            group_id="default",
        )

    assert "extractor_api_error" in str(exc.value)
    assert exc.value.retryable is True
    update_episode_status.assert_not_awaited()
    apply_bootstrap_part_of_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_evidence_projection_executor_commits_and_defers_evidence():
    plan = _plan()
    episode = _episode()
    committed_ev = EvidenceCandidate(
        evidence_id="evi_commit",
        episode_id=episode.id,
        group_id="default",
        fact_class="entity",
        confidence=0.9,
        payload={"name": "Python"},
    )
    deferred_ev = EvidenceCandidate(
        evidence_id="evi_defer",
        episode_id=episode.id,
        group_id="default",
        fact_class="relationship",
        confidence=0.55,
        payload={"source": "Python", "predicate": "USED_FOR", "target": "AI"},
    )
    evidence_bundle = EvidenceBundle(
        episode_id=episode.id,
        group_id="default",
        candidates=[committed_ev, deferred_ev],
    )
    projection_bundle = ProjectionBundle(
        episode_id=episode.id,
        plan=plan,
        entities=[EntityCandidate(name="Python", entity_type="Technology")],
        claims=[],
    )
    graph = AsyncMock()
    graph.get_entity_count = AsyncMock(return_value=12)
    graph.store_evidence = AsyncMock()
    build_evidence_bundle = MagicMock(return_value=evidence_bundle)
    commit_policy = SimpleNamespace(
        evaluate=lambda bundle, _entity_count: [
            CommitDecision(evidence_id=bundle.candidates[0].evidence_id, action="commit"),
            CommitDecision(evidence_id=bundle.candidates[1].evidence_id, action="defer"),
        ],
    )
    update_episode_status = AsyncMock()
    materialize_evidence = AsyncMock(
        return_value=SimpleNamespace(
            apply_outcome=ApplyOutcome(entity_map={"Python": "ent_python"}),
            committed_ids={"evi_commit": "ent_python"},
            bundle=projection_bundle,
        ),
    )

    def serialize_evidence_records(pairs, *, status, commit_reason=None):
        return [
            {
                "evidence_id": evidence.evidence_id,
                "status": status,
                "commit_reason": commit_reason,
            }
            for evidence, _decision in pairs
        ]

    def apply_committed_ids(rows, committed_ids):
        committed_rows = []
        unresolved_rows = []
        for row in rows:
            committed_id = committed_ids.get(row["evidence_id"])
            if committed_id:
                committed_rows.append({**row, "committed_id": committed_id})
            else:
                unresolved_rows.append(row)
        return committed_rows, unresolved_rows

    executor = EvidenceProjectionExecutor(
        graph_store=graph,
        cfg=ActivationConfig(evidence_store_deferred=True),
        build_evidence_bundle=build_evidence_bundle,
        build_adjudication_requests=lambda *_args: [],
        serialize_candidate_records=lambda *_args, **_kwargs: [],
        serialize_evidence_records=serialize_evidence_records,
        materialize_evidence=materialize_evidence,
        apply_committed_ids=apply_committed_ids,
        update_episode_status=update_episode_status,
        commit_policy=commit_policy,
    )

    outcome = await executor.execute(
        episode=episode,
        plan=plan,
        group_id="default",
        model_tier="fast",
    )

    assert outcome.bundle is projection_bundle
    assert outcome.apply_outcome.entity_map == {"Python": "ent_python"}
    assert outcome.entity_map == {"Python": "ent_python"}
    assert outcome.used_evidence_materializer is True
    build_evidence_bundle.assert_called_once_with(
        text=plan.selected_text,
        episode_id=episode.id,
        group_id="default",
        cue=None,
        proposed_entities=None,
        proposed_relationships=None,
        model_tier="fast",
    )
    update_episode_status.assert_any_await(
        episode.id,
        EpisodeStatus.RESOLVING,
        group_id="default",
    )
    materialize_evidence.assert_awaited_once()
    assert graph.store_evidence.await_count == 2
    deferred_call, committed_call = graph.store_evidence.await_args_list
    assert deferred_call.args[0] == [
        {"evidence_id": "evi_defer", "status": "deferred", "commit_reason": None},
    ]
    assert deferred_call.kwargs == {"group_id": "default", "default_status": "deferred"}
    assert committed_call.args[0] == [
        {
            "evidence_id": "evi_commit",
            "status": "committed",
            "commit_reason": "committed_on_hot_path",
            "committed_id": "ent_python",
        },
    ]
    assert committed_call.kwargs == {"group_id": "default", "default_status": "committed"}


@pytest.mark.asyncio
async def test_evidence_projection_executor_persists_rejected_proposal_rows():
    """M0.8 loud reject: rejected client-proposal rows persist with per-row reasons."""
    plan = _plan()
    episode = _episode()
    committed_ev = EvidenceCandidate(
        evidence_id="evi_commit",
        episode_id=episode.id,
        group_id="default",
        fact_class="entity",
        confidence=0.9,
        source_type="client_proposal",
        payload={"name": "Alice"},
    )
    rejected_ev = EvidenceCandidate(
        evidence_id="evi_reject",
        episode_id=episode.id,
        group_id="default",
        fact_class="relationship",
        confidence=0.40,
        source_type="client_proposal",
        payload={"subject": "User", "predicate": "INTERESTED_IN", "object": "Jazz"},
        corroborating_signals=["client_proposal", "predicate_not_allowed"],
    )
    evidence_bundle = EvidenceBundle(
        episode_id=episode.id,
        group_id="default",
        candidates=[committed_ev, rejected_ev],
        extractor_stats={"extraction_path": "client_proposals"},
    )
    projection_bundle = ProjectionBundle(
        episode_id=episode.id,
        plan=plan,
        entities=[EntityCandidate(name="Alice", entity_type="Person")],
        claims=[],
    )
    graph = AsyncMock()
    graph.get_entity_count = AsyncMock(return_value=12)
    graph.store_evidence = AsyncMock()
    commit_policy = SimpleNamespace(
        evaluate=lambda bundle, _entity_count: [
            CommitDecision(evidence_id=bundle.candidates[0].evidence_id, action="commit"),
            CommitDecision(
                evidence_id=bundle.candidates[1].evidence_id,
                action="reject",
                reason="predicate_not_allowed",
            ),
        ],
    )
    materialize_evidence = AsyncMock(
        return_value=SimpleNamespace(
            apply_outcome=ApplyOutcome(entity_map={"Alice": "ent_alice"}),
            committed_ids={"evi_commit": "ent_alice"},
            bundle=projection_bundle,
        ),
    )

    def serialize_evidence_records(pairs, *, status, commit_reason=None):
        return [
            {
                "evidence_id": evidence.evidence_id,
                "status": status,
                "commit_reason": commit_reason,
            }
            for evidence, _decision in pairs
        ]

    def apply_committed_ids(rows, committed_ids):
        committed_rows = []
        unresolved_rows = []
        for row in rows:
            committed_id = committed_ids.get(row["evidence_id"])
            if committed_id:
                committed_rows.append({**row, "committed_id": committed_id})
            else:
                unresolved_rows.append(row)
        return committed_rows, unresolved_rows

    executor = EvidenceProjectionExecutor(
        graph_store=graph,
        cfg=ActivationConfig(evidence_store_deferred=True),
        build_evidence_bundle=MagicMock(return_value=evidence_bundle),
        build_adjudication_requests=lambda *_args: [],
        serialize_candidate_records=lambda *_args, **_kwargs: [],
        serialize_evidence_records=serialize_evidence_records,
        materialize_evidence=materialize_evidence,
        apply_committed_ids=apply_committed_ids,
        update_episode_status=AsyncMock(),
        commit_policy=commit_policy,
    )

    await executor.execute(
        episode=episode,
        plan=plan,
        group_id="default",
    )

    rejected_calls = [
        call
        for call in graph.store_evidence.await_args_list
        if call.kwargs.get("default_status") == "rejected"
    ]
    assert len(rejected_calls) == 1
    assert rejected_calls[0].args[0] == [
        {
            "evidence_id": "evi_reject",
            "status": "rejected",
            "commit_reason": "predicate_not_allowed",
        },
    ]


@pytest.mark.asyncio
async def test_evidence_projection_executor_requires_commit_policy():
    plan = _plan()
    episode = _episode()
    graph = AsyncMock()
    graph.get_entity_count = AsyncMock(return_value=0)
    update_episode_status = AsyncMock()
    executor = EvidenceProjectionExecutor(
        graph_store=graph,
        cfg=ActivationConfig(),
        build_evidence_bundle=lambda **_kwargs: EvidenceBundle(
            episode_id=episode.id,
            group_id="default",
        ),
        build_adjudication_requests=lambda *_args: [],
        serialize_candidate_records=lambda *_args, **_kwargs: [],
        serialize_evidence_records=lambda *_args, **_kwargs: [],
        materialize_evidence=AsyncMock(),
        apply_committed_ids=lambda rows, _ids: (rows, []),
        update_episode_status=update_episode_status,
    )

    with pytest.raises(ProjectionError, match="evidence_commit_policy_missing"):
        await executor.execute(
            episode=episode,
            plan=plan,
            group_id="default",
        )

    update_episode_status.assert_not_awaited()
