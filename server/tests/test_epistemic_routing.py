"""Unit tests for epistemic routing, claim states, and answer contracts."""

from engram.models.epistemic import EvidenceClaim
from engram.retrieval.epistemic import (
    apply_answer_contract_to_evidence_plan,
    build_evidence_plan,
    infer_claim_state,
    reconcile_claims,
    resolve_answer_contract,
    route_question,
    should_materialize_conversation_decision,
)


def test_route_question_remember():
    frame = route_question("my son did great today in soccer")
    assert frame.mode == "remember"
    assert frame.domain == "personal"


def test_route_question_inspect():
    frame = route_question(
        "how do we install the OpenClaw skill?",
        project_path="/tmp/engram",
        surface_capabilities={"workspace_available": True},
    )
    assert frame.mode == "inspect"
    assert frame.requires_workspace is True


def test_route_question_reconcile():
    frame = route_question("what did we decide about launching Engram publicly?")
    assert frame.mode == "reconcile"
    assert frame.timeframe == "both"


def test_answer_contract_compare_for_default_question():
    frame = route_question("is full mode rework by default?")
    plan = build_evidence_plan(frame)

    contract = resolve_answer_contract(
        "is full mode rework by default?",
        frame=frame,
        plan=plan,
    )

    assert contract.operator == "compare"
    assert "raw_default" in contract.relevant_scopes
    assert "install_default" in contract.relevant_scopes
    assert "runtime_current" in contract.relevant_scopes


def test_answer_contract_plan_with_recommendation():
    frame = route_question("How would we approach the OpenClaw install plan? What do you think?")
    plan = build_evidence_plan(frame)

    contract = resolve_answer_contract(
        "How would we approach the OpenClaw install plan? What do you think?",
        frame=frame,
        plan=plan,
    )

    assert contract.operator == "plan"
    assert contract.allow_recommendation is True


def test_answer_contract_recommend():
    frame = route_question("what do you think is the best launch path?")
    plan = build_evidence_plan(frame)

    contract = resolve_answer_contract(
        "what do you think is the best launch path?",
        frame=frame,
        plan=plan,
    )

    assert contract.operator == "recommend"


def test_build_evidence_plan_for_reconcile_uses_multiple_sources():
    frame = route_question("what did we decide about launching Engram publicly?")
    plan = build_evidence_plan(
        frame,
        surface_capabilities={"workspace_available": False},
    )
    assert plan.use_memory is True
    assert plan.use_artifacts is True
    assert plan.use_runtime is True


def test_reconcile_route_requires_artifacts_and_discourages_fact_search():
    question = "what did we decide about launching Engram publicly?"
    frame = route_question(question, project_path="/tmp/engram")
    plan = build_evidence_plan(frame)
    contract = resolve_answer_contract(question, frame=frame, plan=plan)

    plan = apply_answer_contract_to_evidence_plan(
        question,
        frame=frame,
        plan=plan,
        answer_contract=contract,
    )

    assert plan.required_next_sources == ["artifacts"]
    assert "facts" in plan.discouraged_sources
    assert plan.source_queries["artifacts"] == "Engram public launch distribution OpenClaw"


def test_compare_route_requires_artifacts_and_runtime():
    question = "is full mode rework by default?"
    frame = route_question(question, project_path="/tmp/engram")
    plan = build_evidence_plan(frame)
    contract = resolve_answer_contract(question, frame=frame, plan=plan)

    plan = apply_answer_contract_to_evidence_plan(
        question,
        frame=frame,
        plan=plan,
        answer_contract=contract,
    )

    assert plan.required_next_sources == ["artifacts", "runtime"]
    assert plan.source_queries["artifacts"] == "full mode rework default integration profile"


def test_historical_only_reconcile_can_remain_memory_only():
    question = "what did we decide about my weekend plan?"
    frame = route_question(question)
    plan = build_evidence_plan(frame)
    contract = resolve_answer_contract(question, frame=frame, plan=plan)

    plan = apply_answer_contract_to_evidence_plan(
        question,
        frame=frame,
        plan=plan,
        answer_contract=contract,
    )

    assert plan.use_memory is True
    assert plan.required_next_sources == []
    assert plan.use_artifacts is False
    assert plan.use_runtime is False


def test_reconcile_claims_reports_conflict():
    frame = route_question("what did we decide about launching Engram publicly?")
    answer_contract = resolve_answer_contract(
        "what did we decide about launching Engram publicly?",
        frame=frame,
        plan=build_evidence_plan(frame),
    )
    memory_claim = EvidenceClaim(
        subject="Engram",
        predicate="public_launch_path",
        object="OpenClaw",
        source_type="memory",
        authority_type="historical",
        externalization_state="discussed",
        confidence=0.8,
    )
    artifact_claim = EvidenceClaim(
        subject="Engram",
        predicate="public_launch_path",
        object="Direct website launch",
        source_type="artifact",
        authority_type="canonical",
        externalization_state="documented",
        confidence=0.9,
    )
    result = reconcile_claims(
        frame,
        memory_claims=[memory_claim],
        artifact_claims=[artifact_claim],
        runtime_claims=[],
        answer_contract=answer_contract,
    )
    assert result.status == "conflict"
    assert any("Preserve earlier discussion" in hint for hint in result.answer_hints)


def test_resolve_answer_contract_marks_unresolved_when_only_discussed_claims_exist():
    frame = route_question("what did we decide about launching Engram publicly?")
    plan = build_evidence_plan(frame)
    claims = [
        EvidenceClaim(
            subject="Engram",
            predicate="decision_statement",
            object="We were leaning toward OpenClaw for launch.",
            source_type="memory",
            authority_type="historical",
            externalization_state="discussed",
            claim_state="tentative",
            confidence=0.7,
        )
    ]

    contract = resolve_answer_contract(
        "what did we decide about launching Engram publicly?",
        frame=frame,
        plan=plan,
        claims=claims,
    )

    assert contract.operator == "unresolved_state_report"


def test_infer_claim_state_for_memory_tentative_and_decided():
    tentative = EvidenceClaim(
        subject="Engram",
        predicate="decision_statement",
        object="Maybe OpenClaw is the better option.",
        source_type="memory",
        authority_type="historical",
        externalization_state="discussed",
    )
    decided = EvidenceClaim(
        subject="Engram",
        predicate="decision_statement",
        object="We decided the plan is OpenClaw.",
        source_type="memory",
        authority_type="historical",
        externalization_state="discussed",
    )

    assert infer_claim_state(tentative) == "tentative"
    assert infer_claim_state(decided) == "decided"


def test_infer_claim_state_for_artifact_runtime_and_superseded():
    documented = EvidenceClaim(
        subject="Engram",
        predicate="public_launch_path",
        object="OpenClaw",
        source_type="artifact",
        authority_type="canonical",
        externalization_state="documented",
        provenance={"path": "README.md"},
    )
    implemented = EvidenceClaim(
        subject="Engram",
        predicate="integration_profile",
        object="rework",
        source_type="artifact",
        authority_type="current",
        externalization_state="implemented",
        provenance={"path": ".env.example"},
    )
    effective = EvidenceClaim(
        subject="Engram",
        predicate="integration_profile",
        object="rework",
        source_type="runtime",
        authority_type="current",
        externalization_state="effective",
    )
    superseded = EvidenceClaim(
        subject="Engram",
        predicate="SUPERSEDED_BY",
        object="OpenClaw",
        source_type="memory",
        authority_type="historical",
        externalization_state="superseded",
    )

    assert infer_claim_state(documented) == "documented"
    assert infer_claim_state(implemented) == "implemented"
    assert infer_claim_state(effective) == "effective"
    assert infer_claim_state(superseded) == "superseded"


def test_should_materialize_conversation_decision_rejects_recollection_prompts():
    assert not should_materialize_conversation_decision(
        "what did we decide about launching Engram publicly?"
    )
    assert not should_materialize_conversation_decision("OpenClaw ring any bells?")


def test_should_materialize_conversation_decision_accepts_explicit_commitment():
    assert should_materialize_conversation_decision(
        "we decided the plan is to launch Engram through OpenClaw"
    )
