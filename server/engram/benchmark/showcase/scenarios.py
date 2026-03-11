"""Deterministic scenario catalog for the showcase benchmark."""

from __future__ import annotations

import random

from engram.benchmark.showcase.models import (
    AnswerTask,
    BudgetProfile,
    ExtractionSpec,
    ScenarioProbe,
    ScenarioTurn,
    ShowcaseScenario,
    TurnAction,
)

_QUICK_SCENARIOS = {
    "cue_delayed_relevance",
    "temporal_override",
    "prospective_trigger",
    "cross_cluster_association",
}

_SCALE_SCENARIOS = {
    "cue_delayed_relevance",
    "open_loop_recovery",
    "multi_session_continuity",
    "context_budget_compression",
    "selective_extraction_efficiency",
    "summary_drift_resistance",
}


def _entity(
    name: str,
    entity_type: str,
    *,
    summary: str | None = None,
    attributes: dict | None = None,
) -> dict:
    return {
        "name": name,
        "entity_type": entity_type,
        "summary": summary,
        "attributes": attributes or {},
    }


def _relationship(
    source: str,
    target: str,
    predicate: str,
    *,
    polarity: str = "positive",
    weight: float = 1.0,
) -> dict:
    return {
        "source": source,
        "target": target,
        "predicate": predicate,
        "polarity": polarity,
        "weight": weight,
    }


def _answer_task(question: str, gold_answer, *, note: str = "") -> AnswerTask:
    return AnswerTask(
        question=question,
        gold_answer=gold_answer,
        answer_grader="field_match",
        expected_format="json",
        answer_budget_tokens=100,
        note=note,
    )


def _ambient_turns(
    prefix: str,
    count: int,
    seed: int,
    *,
    start_index: int,
    action: TurnAction = "observe",
) -> list[ScenarioTurn]:
    rng = random.Random((seed + 1) * (count + 17) + start_index)
    topics = [
        "lint cleanup",
        "roadmap grooming",
        "docs formatting",
        "pairing notes",
        "incident follow-up",
        "design review",
        "dependency audit",
        "UI polish",
        "meeting recap",
        "release prep",
    ]
    turns: list[ScenarioTurn] = []
    for offset in range(count):
        topic = topics[(offset + rng.randint(0, len(topics) - 1)) % len(topics)]
        turns.append(
            ScenarioTurn(
                id=f"{prefix}_{offset + 1}",
                action=action,
                content=(
                    f"Ambient note {offset + 1}: {topic} for {prefix.replace('_', ' ')}. "
                    f"No durable decisions changed here."
                ),
                source="showcase:ambient",
            )
        )
    return turns


def build_showcase_scenarios(mode: str = "quick", seed: int = 0) -> list[ShowcaseScenario]:
    """Build the deterministic scenario suite for a given mode and seed."""
    history_multiplier = 1
    if mode == "scale":
        history_multiplier = 4

    scenarios = _all_scenarios(seed=seed, history_multiplier=history_multiplier)
    if mode == "quick":
        return [scenario for scenario in scenarios if scenario.id in _QUICK_SCENARIOS]
    if mode == "scale":
        return [scenario for scenario in scenarios if scenario.id in _SCALE_SCENARIOS]
    return scenarios


def _all_scenarios(seed: int, history_multiplier: int) -> list[ShowcaseScenario]:
    return [
        _cue_delayed_relevance(seed, history_multiplier),
        _temporal_override(seed),
        _negation_correction(seed),
        _open_loop_recovery(seed, history_multiplier),
        _prospective_trigger(seed),
        _cross_cluster_association(seed),
        _latent_open_loop_cue(seed, history_multiplier),
        _multi_session_continuity(seed, history_multiplier),
        _context_budget_compression(seed, history_multiplier),
        _meta_contamination_resistance(seed),
        _selective_extraction_efficiency(seed, history_multiplier),
        _correction_chain(seed),
        _summary_drift_resistance(seed, history_multiplier),
    ]


def _cue_delayed_relevance(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="observe_maya_note",
            action="observe",
            content=(
                "Maya mentioned the cedar cache migration needs one extra smoke test "
                "before release."
            ),
            source="showcase:meeting",
        ),
        ScenarioTurn(
            id="observe_maya_followup",
            action="observe",
            content=(
                "Follow-up from Maya: the cedar cache migration still needs that "
                "extra smoke test before release."
            ),
            source="showcase:meeting",
        ),
        ScenarioTurn(
            id="evergreen_distractor",
            action="observe",
            content="Evergreen cache migration already passed every smoke test.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "cue_delayed",
            4 * history_multiplier,
            seed,
            start_index=1,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=120, answer_budget_tokens=90)
    probes = [
        ScenarioProbe(
            id="cue_probe",
            after_turn_index=len(turns) - 1,
            operation="recall",
            query="What did Maya mention about the cedar cache migration?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["cedar cache migration", "extra smoke test"],
            required_evidence_result_types=["cue_episode"],
            expected_result_types=["cue_episode"],
            capability_tags=["cue", "continuity"],
            note="Observed-only memory should still surface later as latent evidence.",
        )
    ]
    answer_task = _answer_task(
        "Summarize Maya's cedar cache reminder as JSON.",
        {"subject": "cedar cache migration", "needed": "one extra smoke test"},
        note="Answer track should reduce the cue recall to the operative fact.",
    )
    return ShowcaseScenario(
        id="cue_delayed_relevance",
        title="Cue Delayed Relevance",
        why_it_matters="Observed content can stay cheap until a later query makes it useful.",
        turns=turns,
        probes=probes,
        capability_tags=["cue", "continuity"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["lexical_overlap", "release_noise"],
    )


def _temporal_override(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="aurora_old",
            action="remember",
            content="Aurora API currently uses api.v1.internal as its base URL.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Aurora API",
                        "Project",
                        summary="Internal API service for Aurora.",
                        attributes={"base_url": "api.v1.internal"},
                    )
                ]
            ),
            source="showcase:temporal",
        ),
        ScenarioTurn(
            id="aurora_distractor",
            action="observe",
            content="Aurora Docs still mention api.v1.internal in archived notes.",
            source="showcase:distractor",
        ),
        ScenarioTurn(
            id="aurora_new",
            action="remember",
            content="Correction: Aurora API now uses api.v2.internal as its base URL.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Aurora API",
                        "Project",
                        summary="Internal API service for Aurora.",
                        attributes={"base_url": "api.v2.internal"},
                    )
                ]
            ),
            source="showcase:temporal",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=80, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="temporal_probe",
            after_turn_index=2,
            operation="get_context",
            topic_hint="Aurora API",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["base_url: api.v2.internal"],
            forbidden_evidence=["api.v1.internal"],
            expected_result_types=["context"],
            disallowed_result_types=["episode", "cue_episode"],
            historical_evidence_allowed=False,
            capability_tags=["temporal"],
            note="Current truth should replace stale truth in compressed context.",
        )
    ]
    answer_task = _answer_task(
        "Return Aurora API's current base URL as JSON.",
        {"base_url": "api.v2.internal"},
    )
    return ShowcaseScenario(
        id="temporal_override",
        title="Temporal Override",
        why_it_matters="Engram should preserve the latest fact instead of surfacing both versions.",
        turns=turns,
        probes=probes,
        capability_tags=["temporal"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["stale_docs", "archived_history"],
    )


def _negation_correction(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="falcon_react",
            action="remember",
            content="Falcon Dashboard uses React for its frontend.",
            extraction=ExtractionSpec(
                entities=[
                    _entity("Falcon Dashboard", "Project", summary="Customer analytics UI."),
                    _entity("React", "Technology", summary="Frontend framework."),
                ],
                relationships=[
                    _relationship("Falcon Dashboard", "React", "USES"),
                ],
            ),
            source="showcase:negation",
        ),
        ScenarioTurn(
            id="harrier_react",
            action="observe",
            content="Harrier Dashboard still uses React for its frontend.",
            source="showcase:distractor",
        ),
        ScenarioTurn(
            id="falcon_svelte",
            action="remember",
            content="Falcon Dashboard stopped using React and now uses Svelte.",
            extraction=ExtractionSpec(
                entities=[
                    _entity("Falcon Dashboard", "Project", summary="Customer analytics UI."),
                    _entity("React", "Technology", summary="Frontend framework."),
                    _entity("Svelte", "Technology", summary="Frontend framework."),
                ],
                relationships=[
                    _relationship("Falcon Dashboard", "React", "USES", polarity="negative"),
                    _relationship("Falcon Dashboard", "Svelte", "USES"),
                ],
            ),
            source="showcase:negation",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=120, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="negation_probe",
            after_turn_index=2,
            operation="recall",
            query="Which framework does Falcon Dashboard use now?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["USES Svelte"],
            forbidden_evidence=["USES React"],
            expected_result_types=["entity"],
            allowed_result_types=["entity", "context"],
            disallowed_result_types=["episode", "cue_episode"],
            historical_evidence_allowed=False,
            capability_tags=["temporal", "negation"],
            note="Invalidated relationships should not keep surfacing as current truth.",
        )
    ]
    answer_task = _answer_task(
        "Return Falcon Dashboard's current frontend framework as JSON.",
        {"framework": "Svelte"},
    )
    return ShowcaseScenario(
        id="negation_correction",
        title="Negation And Correction",
        why_it_matters=(
            "Negative polarity should suppress stale relationships instead of appending noise."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["temporal", "negation"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["neighbor_entity_overlap", "stale_framework"],
    )


def _open_loop_recovery(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="staging_loop",
            action="observe",
            content="Open loop: rotate staging secrets before Friday.",
            source="showcase:loop",
        ),
        ScenarioTurn(
            id="production_done",
            action="observe",
            content="Closed loop: rotate production secrets completed yesterday.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "open_loop",
            5 * history_multiplier,
            seed,
            start_index=20,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=110, answer_budget_tokens=90)
    probes = [
        ScenarioProbe(
            id="open_loop_probe",
            after_turn_index=len(turns) - 1,
            operation="recall",
            query="What is still open for staging?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["rotate staging secrets before Friday"],
            forbidden_evidence=["production secrets completed"],
            expected_result_types=["cue_episode", "episode"],
            capability_tags=["open_loop"],
            note="The system should resurface unfinished work when related context returns.",
        )
    ]
    answer_task = _answer_task(
        "Return the remaining staging open loop as JSON.",
        {"open_loop": "rotate staging secrets before Friday"},
    )
    return ShowcaseScenario(
        id="open_loop_recovery",
        title="Open Loop Recovery",
        why_it_matters=(
            "Latent unresolved work should return later without keeping full history in prompt."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["open_loop"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["closed_loop_trap", "session_noise"],
    )


def _prospective_trigger(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="auth_module_memory",
            action="remember",
            content="Auth Module owns JWT token validation.",
            extraction=ExtractionSpec(
                entities=[
                    _entity("Auth Module", "Project", summary="Authentication subsystem."),
                    _entity(
                        "JWT Token Validation",
                        "Concept",
                        summary="Validation path for access tokens.",
                    ),
                ],
                relationships=[
                    _relationship("JWT Token Validation", "Auth Module", "PART_OF"),
                ],
            ),
            source="showcase:prospective",
        ),
        ScenarioTurn(
            id="billing_intention",
            action="intend",
            trigger_text="Billing release guard",
            action_text="Verify CSV export before deploying",
            entity_names=["Billing Module"],
            threshold=0.1,
            priority="normal",
        ),
        ScenarioTurn(
            id="auth_intention",
            action="intend",
            trigger_text="Auth module release guard",
            action_text="Check XSS fix before deploying",
            entity_names=["Auth Module"],
            threshold=0.1,
            priority="high",
        ),
        ScenarioTurn(
            id="jwt_work",
            action="remember",
            content="JWT token validation is on today's agenda.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "JWT Token Validation",
                        "Concept",
                        summary="Validation path for access tokens.",
                    ),
                    _entity("Auth Module", "Project", summary="Authentication subsystem."),
                ],
                relationships=[
                    _relationship("JWT Token Validation", "Auth Module", "PART_OF"),
                ],
            ),
            source="showcase:prospective",
        ),
        ScenarioTurn(
            id="dismiss_auth_intention",
            action="dismiss_intention",
            ref="auth_intention",
            hard_delete=True,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=120, answer_budget_tokens=80)
    probes = [
        ScenarioProbe(
            id="prospective_probe_before_dismiss",
            after_turn_index=3,
            operation="recall",
            query="What should I remember while working on JWT token validation?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["Check XSS fix before deploying"],
            forbidden_evidence=["Verify CSV export before deploying"],
            expected_result_types=["entity"],
            capability_tags=["prospective"],
            note="Related entity activity should surface the linked intention.",
        ),
        ScenarioProbe(
            id="prospective_probe_after_dismiss",
            after_turn_index=4,
            operation="recall",
            query="What should I remember while working on JWT token validation?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            forbidden_evidence=["Check XSS fix before deploying"],
            capability_tags=["prospective"],
            note="Dismissed intentions should stop resurfacing.",
        ),
    ]
    answer_task = _answer_task(
        "Return the operative JWT reminder as JSON.",
        {"reminder": "Check XSS fix before deploying"},
    )
    return ShowcaseScenario(
        id="prospective_trigger",
        title="Prospective Trigger",
        why_it_matters=(
            "Intentions should fire from related entity activity"
            " rather than raw lexical overlap alone."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["prospective"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["unrelated_intention", "dismissal_regression"],
    )


def _cross_cluster_association(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="bearer_sentinel_mesh",
            action="remember",
            content="Bearer Validation is part of Sentinel Mesh.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Bearer Validation",
                        "Capability",
                        summary="Validation path for bearer credentials.",
                    ),
                    _entity("Sentinel Mesh", "Project", summary="Internal security routing mesh."),
                ],
                relationships=[
                    _relationship("Bearer Validation", "Sentinel Mesh", "PART_OF"),
                ],
            ),
            source="showcase:graph",
        ),
        ScenarioTurn(
            id="public_docs_distractor",
            action="observe",
            content="June Window is the documentation review slot for public site copy.",
            source="showcase:distractor",
        ),
        ScenarioTurn(
            id="sentinel_window",
            action="remember",
            content="Sentinel Mesh routes through Launch Ring for critical releases.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Launch Ring",
                        "Concept",
                        summary="An internal routing gate used by infrastructure teams.",
                    ),
                    _entity("Sentinel Mesh", "Project", summary="Internal security routing mesh."),
                ],
                relationships=[
                    _relationship("Sentinel Mesh", "Launch Ring", "ROUTES_THROUGH"),
                ],
            ),
            source="showcase:graph",
        ),
        ScenarioTurn(
            id="launch_ring_blocker",
            action="remember",
            content="Launch Ring requires Blue 17 before shipping.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Blue 17",
                        "Concept",
                        summary="An internal holdpoint used by infrastructure teams.",
                    ),
                    _entity(
                        "Launch Ring",
                        "Concept",
                        summary="An internal routing gate used by infrastructure teams.",
                    ),
                ],
                relationships=[
                    _relationship("Launch Ring", "Blue 17", "REQUIRES"),
                ],
            ),
            source="showcase:graph",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=120, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="graph_probe",
            after_turn_index=3,
            operation="recall",
            query="What hidden blocker matters for bearer validation work?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["Blue 17"],
            required_evidence_result_types=["entity"],
            expected_result_types=["entity"],
            capability_tags=["association", "graph"],
            note=(
                "Graph spreading should connect lexically distant but structurally linked entities."
            ),
        )
    ]
    answer_task = _answer_task(
        "Return the hidden checkpoint for bearer validation work as JSON.",
        {"dependency": "Blue 17"},
    )
    return ShowcaseScenario(
        id="cross_cluster_association",
        title="Cross Cluster Association",
        why_it_matters=(
            "Graph-aware retrieval should outperform flat lexical retrieval on associative queries."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["association", "graph"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["lexical_neighbor", "two_hop_dependency"],
    )


def _latent_open_loop_cue(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="canary_open_loop",
            action="observe",
            content=(
                "Rina mentioned the canary rollout still needs canary keys rotated before Tuesday."
            ),
            source="showcase:meeting",
        ),
        ScenarioTurn(
            id="canary_open_loop_followup",
            action="observe",
            content=(
                "Follow-up from Rina: the canary rollout still needs"
                " canary keys rotated before Tuesday."
            ),
            source="showcase:meeting",
        ),
        ScenarioTurn(
            id="canary_closed_distractor",
            action="observe",
            content="Milo confirmed the staging checklist already finished yesterday.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "latent_canary_loop",
            4 * history_multiplier,
            seed,
            start_index=150,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=110, answer_budget_tokens=80)
    probes = [
        ScenarioProbe(
            id="latent_open_loop_probe",
            after_turn_index=len(turns) - 1,
            operation="recall",
            query="What did Rina mention about the canary rollout?",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["canary keys", "before Tuesday"],
            required_evidence_result_types=["cue_episode"],
            expected_result_types=["cue_episode"],
            capability_tags=["cue", "open_loop"],
            note=(
                "A latent unresolved task should come back as cue"
                " evidence instead of raw log replay."
            ),
        ),
    ]
    answer_task = _answer_task(
        "Return Rina's canary rollout reminder as JSON.",
        {"open_loop": "rotate canary keys before Tuesday"},
    )
    return ShowcaseScenario(
        id="latent_open_loop_cue",
        title="Latent Open Loop Cue",
        why_it_matters=(
            "Unfinished work should resurface through latent cue recall,"
            " not only through exact lexical search."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["cue", "open_loop"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["latent_open_loop", "closed_loop_overlap"],
    )


def _multi_session_continuity(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="northstar_codename",
            action="remember",
            session_id="session-a",
            content="Project Northstar's codename is Atlas.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Project Northstar",
                        "Project",
                        summary="Internal planning initiative.",
                        attributes={"codename": "Atlas"},
                    )
                ]
            ),
            source="showcase:continuity",
        ),
        ScenarioTurn(
            id="southstar_distractor",
            action="observe",
            session_id="session-b",
            content="Project Southstar's codename is Vector.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "northstar_recent",
            4 * history_multiplier,
            seed,
            start_index=40,
            action="remember",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=70, answer_budget_tokens=60)
    probes = [
        ScenarioProbe(
            id="continuity_probe",
            after_turn_index=len(turns) - 1,
            operation="get_context",
            topic_hint="Project Northstar",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["codename: Atlas"],
            expected_result_types=["context"],
            capability_tags=["continuity"],
            note="A later session should still inherit the durable project identity fact.",
        )
    ]
    answer_task = _answer_task(
        "Return Project Northstar's codename as JSON.",
        {"codename": "Atlas"},
    )
    return ShowcaseScenario(
        id="multi_session_continuity",
        title="Multi Session Continuity",
        why_it_matters=(
            "Durable project state should survive beyond the immediate conversation window."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["continuity"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["cross_session_name_overlap"],
    )


def _context_budget_compression(seed: int, history_multiplier: int) -> ShowcaseScenario:
    long_note = (
        "Payments migration status: the kickoff covered four vendor meetings, "
        "a rollback rehearsal, three dependency tickets, and a long implementation "
        "discussion about edge retries, but the durable facts are that Priya owns "
        "the migration, the deadline is April 17, and the rollback path is Stripe fallback."
    )
    turns = [
        ScenarioTurn(
            id="payments_long_note",
            action="remember",
            content=long_note,
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Payments Migration",
                        "Project",
                        summary="Cutover plan for the payments stack.",
                        attributes={
                            "owner": "Priya",
                            "deadline": "April 17",
                            "rollback": "Stripe fallback",
                        },
                    )
                ]
            ),
            source="showcase:compression",
        ),
        ScenarioTurn(
            id="payments_distractor",
            action="observe",
            content="Payments experiments owner is Aria and their deadline is May 2.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "payments_context",
            3 * history_multiplier,
            seed,
            start_index=60,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=60, answer_budget_tokens=80)
    probes = [
        ScenarioProbe(
            id="compression_probe",
            after_turn_index=len(turns) - 1,
            operation="get_context",
            topic_hint="Payments Migration",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["owner: Priya", "deadline: April 17"],
            forbidden_evidence=["owner is Aria", "May 2"],
            expected_result_types=["context"],
            capability_tags=["compression"],
            note="Structured context should preserve key facts inside a tighter token budget.",
        )
    ]
    answer_task = _answer_task(
        "Return the payments migration owner and deadline as JSON.",
        {"owner": "Priya", "deadline": "April 17"},
    )
    return ShowcaseScenario(
        id="context_budget_compression",
        title="Context Budget Compression",
        why_it_matters=(
            "Structured memory should keep the key facts even when raw notes get truncated."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["compression"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["owner_overlap", "deadline_overlap"],
    )


def _meta_contamination_resistance(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="konner_style",
            action="remember",
            content="Alex prefers concise writeups.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Alex",
                        "Person",
                        summary="Primary user profile.",
                        attributes={"writing_style": "concise writeups"},
                    )
                ]
            ),
            source="showcase:meta",
        ),
        ScenarioTurn(
            id="konner_debug_noise",
            action="observe",
            content="DEBUG ONLY: Alex activation score 0.91, pipeline=green, entity id ent_123.",
            source="showcase:meta",
        ),
        ScenarioTurn(
            id="konner_summary_noise",
            action="observe",
            content=(
                "Assistant scratchpad: maybe Alex likes"
                " long-form status narratives? confidence=0.12"
            ),
            source="showcase:meta",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=80, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="meta_probe",
            after_turn_index=2,
            operation="get_context",
            topic_hint="Alex",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["writing_style: concise writeups"],
            forbidden_evidence=["0.91", "ent_123", "long-form status narratives"],
            expected_result_types=["context"],
            capability_tags=["meta"],
            note="System chatter should not pollute the durable user model.",
        )
    ]
    answer_task = _answer_task(
        "Return Alex's writing style preference as JSON.",
        {"writing_style": "concise writeups"},
    )
    return ShowcaseScenario(
        id="meta_contamination_resistance",
        title="Meta Contamination Resistance",
        why_it_matters="System telemetry must not be mistaken for user memory.",
        turns=turns,
        probes=probes,
        capability_tags=["meta"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["debug_noise", "summary_scratchpad"],
    )


def _selective_extraction_efficiency(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="sparrow_board_note",
            action="observe",
            content=(
                "Board decision: Sparrow Feature should sunset next quarter after the migration."
            ),
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Sparrow Feature",
                        "Project",
                        summary="Legacy feature under review.",
                        attributes={"decision": "sunset next quarter"},
                    )
                ]
            ),
            source="showcase:efficiency",
        ),
        ScenarioTurn(
            id="finch_board_note",
            action="observe",
            content="Board decision: Finch Feature should expand next quarter after the launch.",
            source="showcase:distractor",
        ),
        *_ambient_turns(
            "sparrow_ambient",
            5 * history_multiplier,
            seed,
            start_index=80,
        ),
        ScenarioTurn(
            id="sparrow_project",
            action="project",
            ref="sparrow_board_note",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=70, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="efficiency_probe",
            after_turn_index=len(turns) - 1,
            operation="get_context",
            topic_hint="Sparrow Feature",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["decision: sunset next quarter"],
            forbidden_evidence=["expand next quarter"],
            expected_result_types=["context"],
            capability_tags=["efficiency"],
            note="Only a subset of observed content should need full projection to answer later.",
        )
    ]
    answer_task = _answer_task(
        "Return Sparrow Feature's board decision as JSON.",
        {"decision": "sunset next quarter"},
    )
    return ShowcaseScenario(
        id="selective_extraction_efficiency",
        title="Selective Extraction Efficiency",
        why_it_matters=(
            "Engram should answer later questions without projecting every observed turn."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["efficiency"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["board_overlap", "projection_selectivity"],
    )


def _correction_chain(seed: int) -> ShowcaseScenario:
    _ = seed
    turns = [
        ScenarioTurn(
            id="orbit_red",
            action="remember",
            content="Orbit rollout status is red.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Orbit Rollout",
                        "Project",
                        summary="Deployment track for Orbit.",
                        attributes={"status": "red"},
                    )
                ]
            ),
            source="showcase:correction_chain",
        ),
        ScenarioTurn(
            id="orbit_yellow",
            action="remember",
            content="Correction: Orbit rollout status is yellow.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Orbit Rollout",
                        "Project",
                        summary="Deployment track for Orbit.",
                        attributes={"status": "yellow"},
                    )
                ]
            ),
            source="showcase:correction_chain",
        ),
        ScenarioTurn(
            id="orbit_green",
            action="remember",
            content="Correction: Orbit rollout status is green.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Orbit Rollout",
                        "Project",
                        summary="Deployment track for Orbit.",
                        attributes={"status": "green"},
                    )
                ]
            ),
            source="showcase:correction_chain",
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=70, answer_budget_tokens=60)
    probes = [
        ScenarioProbe(
            id="correction_chain_probe",
            after_turn_index=2,
            operation="get_context",
            topic_hint="Orbit Rollout",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["status: green"],
            forbidden_evidence=["status: red", "status: yellow"],
            expected_result_types=["context"],
            disallowed_result_types=["episode", "cue_episode"],
            historical_evidence_allowed=False,
            capability_tags=["temporal", "negation"],
            note="Only the latest valid correction should survive a chain of updates.",
        )
    ]
    answer_task = _answer_task(
        "Return Orbit rollout's current status as JSON.",
        {"status": "green"},
    )
    return ShowcaseScenario(
        id="correction_chain",
        title="Correction Chain",
        why_it_matters="Repeated corrections should converge to a single current truth.",
        turns=turns,
        probes=probes,
        capability_tags=["temporal", "negation"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["multi_update_chain"],
    )


def _summary_drift_resistance(seed: int, history_multiplier: int) -> ShowcaseScenario:
    turns = [
        ScenarioTurn(
            id="launch_memo_style",
            action="remember",
            content="Launch Memo should stay concise and direct.",
            extraction=ExtractionSpec(
                entities=[
                    _entity(
                        "Launch Memo",
                        "Document",
                        summary="Canonical launch communication style.",
                        attributes={"style": "concise and direct"},
                    )
                ]
            ),
            source="showcase:drift",
        ),
        ScenarioTurn(
            id="launch_memo_paraphrase",
            action="observe",
            content="Paraphrase: the launch memo should be brief and direct.",
            source="showcase:drift",
        ),
        ScenarioTurn(
            id="launch_memo_scratchpad",
            action="observe",
            content="Scratchpad only: maybe a long-form narrative would be interesting someday.",
            source="showcase:drift",
        ),
        *_ambient_turns(
            "memo_drift",
            3 * history_multiplier,
            seed,
            start_index=120,
        ),
    ]
    budget = BudgetProfile(retrieval_limit=5, evidence_max_tokens=80, answer_budget_tokens=70)
    probes = [
        ScenarioProbe(
            id="summary_drift_probe",
            after_turn_index=len(turns) - 1,
            operation="get_context",
            topic_hint="Launch Memo",
            limit=budget.retrieval_limit,
            max_tokens=budget.evidence_max_tokens,
            required_evidence=["style: concise and direct"],
            forbidden_evidence=["long-form narrative"],
            expected_result_types=["context"],
            disallowed_result_types=["episode"],
            historical_evidence_allowed=False,
            capability_tags=["meta", "temporal"],
            note=(
                "Paraphrases and exploratory notes should not distort the canonical current style."
            ),
        )
    ]
    answer_task = _answer_task(
        "Return the Launch Memo style as JSON.",
        {"style": "concise and direct"},
    )
    return ShowcaseScenario(
        id="summary_drift_resistance",
        title="Summary Drift Resistance",
        why_it_matters=(
            "Repeated paraphrases and exploratory chatter should not rewrite current truth."
        ),
        turns=turns,
        probes=probes,
        capability_tags=["meta", "temporal"],
        answer_task=answer_task,
        gold_answer=answer_task.gold_answer,
        answer_grader=answer_task.answer_grader,
        budget_profile=budget,
        distractor_tags=["paraphrase_noise", "exploratory_chatter"],
    )
