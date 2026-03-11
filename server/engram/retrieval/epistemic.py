"""Deterministic epistemic routing and evidence reconciliation."""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import PurePosixPath

from engram.models.epistemic import (
    AnswerContract,
    EpistemicBundle,
    EvidenceClaim,
    EvidencePlan,
    QuestionFrame,
    ReconciliationResult,
)

_PERSONAL_PATTERNS = re.compile(
    r"\b("
    r"my son|my daughter|my wife|my husband|my partner|my kid|my child|"
    r"my mom|my dad|my family|my therapist|my coach|my team|"
    r"i feel|i'm feeling|i was feeling|i get nervous|still nervous|"
    r"he did|she did|they did|talked to|my project"
    r")\b",
    re.IGNORECASE,
)
_PROJECT_PATTERNS = re.compile(
    r"\b("
    r"engram|project|repo|repository|codebase|readme|skill|mcp|openclaw|"
    r"launch|release|roadmap|integration|distribution|plan|strategy|"
    r"feature|bug|config|setup|install|runtime|full mode|rework"
    r")\b",
    re.IGNORECASE,
)
_RUNTIME_PATTERNS = re.compile(
    r"\b("
    r"enabled|default|currently|current|runtime|effective|actual|configured|"
    r"mode|profile|flag|env|environment|setting|settings|implementation"
    r")\b",
    re.IGNORECASE,
)
_INSPECT_PATTERNS = re.compile(
    r"\b("
    r"how do i|how do we|how to|install|configure|setup|set up|"
    r"what is enabled|what's enabled|what does .* default|"
    r"is .* by default|where is|which file|what does the repo say|"
    r"how does .* work|show me the current|what is the current"
    r")\b",
    re.IGNORECASE,
)
_RECONCILE_PATTERNS = re.compile(
    r"\b("
    r"what did we decide|did we decide|what was the call|where did we land|"
    r"i can't remember|i cant remember|can't remember|cant remember|"
    r"decision|decided|plan|strategy|launch|public|distribution|"
    r"roadmap|integration|ship this through|did we ever agree"
    r")\b",
    re.IGNORECASE,
)
_CURRENT_PATTERNS = re.compile(
    r"\b(now|current|currently|today|enabled|default|actual|implemented|runtime)\b",
    re.IGNORECASE,
)
_HISTORICAL_PATTERNS = re.compile(
    r"\b(last time|earlier|before|previously|did we|what did we|remember|decide|said)\b",
    re.IGNORECASE,
)
_INSTALL_PATTERNS = re.compile(
    r"\b(install|setup|set up|configure|env|mcp|skill|serve|docker|run)\b",
    re.IGNORECASE,
)
_DECISION_PATTERNS = re.compile(
    r"\b(decide|decided|decision|plan|strategy|launch|distribution|ship|public|federation)\b",
    re.IGNORECASE,
)
_OPENCLAW_PATTERNS = re.compile(r"\bopenclaw|clawhub\b", re.IGNORECASE)
_FULL_MODE_REWORK_PATTERNS = re.compile(
    r"\bfull mode\b.*\brework\b|\brework\b.*\bfull mode\b",
    re.IGNORECASE,
)
_PLAN_PATTERNS = re.compile(
    r"\b(plan|steps|rollout|approach|next steps|how would we approach|implementation plan)\b",
    re.IGNORECASE,
)
_RECOMMEND_PATTERNS = re.compile(
    r"\b(what do you think|should we|best way|recommend|recommendation|advice)\b",
    re.IGNORECASE,
)
_COMPARE_PATTERNS = re.compile(
    r"\b(by default|out of the box|vs|versus|difference|compare|default posture)\b",
    re.IGNORECASE,
)
_TIMELINE_PATTERNS = re.compile(
    r"\b(when|timeline|history|since|how did this change|evolution|changed over time)\b",
    re.IGNORECASE,
)
_SETTLED_DECISION_PATTERNS = re.compile(
    r"\b(what did we decide|where did we land|what was the call|did we ever agree|final call)\b",
    re.IGNORECASE,
)
_RECOLLECTION_PROMPTS = re.compile(
    r"\b("
    r"what did we decide|did we decide|what was the call|where did we land|"
    r"i can't remember|i cant remember|can't remember|cant remember|"
    r"ring any bells|do you remember|remember when"
    r")\b",
    re.IGNORECASE,
)
_REPO_SCOPE_PATTERNS = re.compile(
    r"\b("
    r"repo|repository|readme|docs|documented|skill|install|setup|config|"
    r"default|launch|launching|public|publicly|distribution|integration|openclaw|release|roadmap"
    r")\b",
    re.IGNORECASE,
)
_TENTATIVE_MARKERS = re.compile(
    r"\b(maybe|probably|leaning|considering|option|candidate|might|could)\b",
    re.IGNORECASE,
)
_DECIDED_MARKERS = re.compile(
    (
        r"\b(decided|going with|priority|default|plan is|will|"
        r"we're shipping|we are shipping|the plan)\b"
    ),
    re.IGNORECASE,
)
_PROFILE_PATTERNS = (
    (
        re.compile(r"\bintegration_profile\b.*\brework\b", re.IGNORECASE),
        "integration_profile",
        "rework",
    ),
    (re.compile(r"\brecall_profile\b.*\ball\b", re.IGNORECASE), "recall_profile", "all"),
    (
        re.compile(r"\bconsolidation_profile\b.*\bstandard\b", re.IGNORECASE),
        "consolidation_profile",
        "standard",
    ),
)
_ENV_ASSIGNMENT = re.compile(r"^(ENGRAM_[A-Z0-9_]+)\s*=\s*(.+)$")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_BULLET = re.compile(r"^[-*]\s+(.+)$")
_JSON_STRING = re.compile(r'^"([^"]+)"\s*:\s*"([^"]+)"')
_TOML_ASSIGNMENT = re.compile(r"^([A-Za-z0-9_.-]+)\s*=\s*[\"']?([^\"']+)[\"']?$")


@dataclass
class _RouteSample:
    mode: str
    operator: str = "direct_answer"
    scopes: tuple[str, ...] = ()


@dataclass
class _ExecutionSample:
    status: str
    operator: str = "direct_answer"
    scopes: tuple[str, ...] = ()
    sources_used: tuple[str, ...] = ()
    artifact_stale_miss: bool = False
    insufficient_after_full_plan: bool = False
    memory_only_false_negative: bool = False


@dataclass
class _RoutingState:
    routes: deque[_RouteSample]
    executions: deque[_ExecutionSample]


class EpistemicRoutingController:
    """In-memory telemetry for routing and reconciliation behavior."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._states: dict[str, _RoutingState] = {}

    def record_route(
        self,
        group_id: str,
        mode: str,
        *,
        operator: str = "direct_answer",
        scopes: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._get_state(group_id).routes.append(
            _RouteSample(
                mode=mode,
                operator=operator,
                scopes=tuple(scopes or ()),
            )
        )

    def record_execution(
        self,
        group_id: str,
        reconciliation: ReconciliationResult,
        plan: EvidencePlan,
        *,
        answer_contract: AnswerContract | None = None,
        artifact_stale_miss: bool = False,
    ) -> None:
        sources_used = tuple(sorted(set(reconciliation.sources_used)))
        self._get_state(group_id).executions.append(
            _ExecutionSample(
                status=reconciliation.status,
                operator=(
                    answer_contract.operator if answer_contract is not None else "direct_answer"
                ),
                scopes=tuple(answer_contract.relevant_scopes) if answer_contract else (),
                sources_used=sources_used,
                artifact_stale_miss=artifact_stale_miss,
                insufficient_after_full_plan=(
                    reconciliation.status == "insufficient"
                    and any(
                        (
                            plan.use_memory,
                            plan.use_artifacts,
                            plan.use_runtime,
                            plan.use_implementation,
                        )
                    )
                ),
                memory_only_false_negative=(
                    reconciliation.status == "memory_only"
                    and (plan.use_artifacts or plan.use_runtime or plan.use_implementation)
                ),
            )
        )

    def snapshot(self, group_id: str) -> dict:
        state = self._get_state(group_id)
        route_counts = Counter(sample.mode for sample in state.routes)
        operator_counts = Counter(sample.operator for sample in state.routes)
        scope_usage = Counter(
            scope
            for sample in state.routes
            for scope in sample.scopes
        )
        execution_counts = Counter(sample.status for sample in state.executions)
        source_counts = Counter(
            source
            for sample in state.executions
            for source in sample.sources_used
        )
        executed_operator_counts = Counter(
            sample.operator for sample in state.executions
        )
        return {
            "route_counts": dict(route_counts),
            "operator_counts": dict(operator_counts),
            "scope_usage": dict(scope_usage),
            "source_usage": dict(source_counts),
            "status_counts": dict(execution_counts),
            "conflict_count": execution_counts["conflict"],
            "unresolved_count": executed_operator_counts["unresolved_state_report"],
            "compare_count": operator_counts["compare"],
            "recommend_count": operator_counts["recommend"],
            "plan_count": operator_counts["plan"],
            "artifact_stale_misses": sum(
                1 for sample in state.executions if sample.artifact_stale_miss
            ),
            "insufficient_after_full_plan": sum(
                1 for sample in state.executions if sample.insufficient_after_full_plan
            ),
            "memory_only_false_negatives": sum(
                1 for sample in state.executions if sample.memory_only_false_negative
            ),
        }

    def _get_state(self, group_id: str) -> _RoutingState:
        state = self._states.get(group_id)
        window = max(50, int(getattr(self._cfg, "recall_need_threshold_window", 100)))
        if state is None:
            state = _RoutingState(
                routes=deque(maxlen=window),
                executions=deque(maxlen=window),
            )
            self._states[group_id] = state
            return state
        if state.routes.maxlen != window:
            state.routes = deque(state.routes, maxlen=window)
        if state.executions.maxlen != window:
            state.executions = deque(state.executions, maxlen=window)
        return state


def route_question(
    question: str,
    *,
    memory_need=None,
    recent_turns: list[str] | None = None,
    project_path: str | None = None,
    surface_capabilities: dict[str, bool] | None = None,
) -> QuestionFrame:
    """Classify the question into remember, inspect, or reconcile."""
    text = " ".join(question.strip().split())
    lowered = text.lower()
    recent_turns = recent_turns or []
    surface_capabilities = surface_capabilities or {}

    personal = bool(_PERSONAL_PATTERNS.search(text))
    project = bool(_PROJECT_PATTERNS.search(text)) or bool(project_path)
    runtime = bool(_RUNTIME_PATTERNS.search(text))
    inspect = bool(_INSPECT_PATTERNS.search(text))
    reconcile = bool(_RECONCILE_PATTERNS.search(text))
    historical = bool(_HISTORICAL_PATTERNS.search(text))
    current = bool(_CURRENT_PATTERNS.search(text))
    install = bool(_INSTALL_PATTERNS.search(text))

    mode = "remember"
    reason = "personal continuity or omitted prior context"
    confidence = 0.68

    if reconcile or (project and historical):
        mode = "reconcile"
        reason = "historical project decision or plan that may now be externalized"
        confidence = 0.84
    elif inspect or runtime or (project and current) or install:
        mode = "inspect"
        reason = "current project or runtime truth should exist in artifacts or config"
        confidence = 0.8
    elif memory_need is not None and getattr(memory_need, "should_recall", False):
        mode = "remember"
        reason = "memory-need analyzer found prior-context value"
        confidence = max(confidence, float(getattr(memory_need, "confidence", 0.0) or 0.0))

    domain = "personal"
    if project and (
        personal
        or getattr(memory_need, "need_type", "") in {"project_state", "open_loop"}
    ):
        domain = "mixed"
    elif runtime:
        domain = "runtime"
    elif project and any(
        token in lowered
        for token in ("launch", "distribution", "public", "skill", "integration")
    ):
        domain = "product"
    elif project:
        domain = "project"
    elif personal:
        domain = "personal"

    timeframe = "historical"
    if current and historical:
        timeframe = "both"
    elif mode == "reconcile":
        timeframe = "both"
    elif mode == "inspect":
        timeframe = "current"
    elif current:
        timeframe = "current"

    expected_authorities: list[str]
    expected_sources: list[str]
    requires_workspace = False
    if mode == "remember":
        expected_authorities = ["personal", "historical"]
        expected_sources = ["memory"]
    elif mode == "inspect":
        expected_authorities = ["current", "canonical"]
        expected_sources = ["artifacts", "runtime"]
        requires_workspace = bool(surface_capabilities.get("workspace_available") and install)
    else:
        expected_authorities = ["historical", "current"]
        expected_sources = ["memory", "artifacts", "runtime"]
        requires_workspace = bool(
            surface_capabilities.get("workspace_available")
            and (install or runtime or "repo" in lowered or "codebase" in lowered)
        )

    if recent_turns and mode == "remember" and not personal and project:
        domain = "mixed"
        confidence = max(confidence, 0.72)

    return QuestionFrame(
        mode=mode,
        domain=domain,
        timeframe=timeframe,
        expected_authorities=expected_authorities,
        expected_sources=expected_sources,
        requires_workspace=requires_workspace,
        confidence=min(0.99, round(confidence, 4)),
        reason=reason,
    )


def build_evidence_plan(
    frame: QuestionFrame,
    *,
    surface_capabilities: dict[str, bool] | None = None,
    cfg=None,
) -> EvidencePlan:
    """Choose evidence sources deterministically from the question frame."""
    caps = dict(surface_capabilities or {})
    artifacts_enabled = bool(getattr(cfg, "artifact_recall_enabled", True))
    runtime_enabled = bool(getattr(cfg, "epistemic_runtime_executor_enabled", True))
    reconcile_enabled = bool(getattr(cfg, "epistemic_reconcile_enabled", True))

    use_memory = frame.mode in {"remember", "reconcile"}
    use_artifacts = artifacts_enabled and frame.mode in {"inspect", "reconcile"}
    use_runtime = runtime_enabled and frame.domain in {"runtime", "project", "product", "mixed"}
    use_implementation = bool(
        caps.get("workspace_available")
        and frame.requires_workspace
        and frame.mode in {"inspect", "reconcile"}
    )

    if frame.mode == "remember":
        use_artifacts = False
        use_runtime = False
        use_implementation = False
    elif frame.mode == "inspect":
        use_memory = frame.timeframe == "both"
    elif not reconcile_enabled:
        use_memory = use_memory and frame.mode == "remember"
        use_artifacts = use_artifacts and frame.mode == "inspect"

    recommended: list[str] = []
    if use_memory:
        recommended.append("memory")
    if use_artifacts:
        recommended.append("artifacts")
    if use_runtime:
        recommended.append("runtime")
    if use_implementation:
        recommended.append("workspace")

    return EvidencePlan(
        use_memory=use_memory,
        use_artifacts=use_artifacts,
        use_implementation=use_implementation,
        use_runtime=use_runtime,
        memory_budget=5 if use_memory else 0,
        artifact_budget=5 if use_artifacts else 0,
        implementation_budget=3 if use_implementation else 0,
        runtime_budget=1 if use_runtime else 0,
        surface_capabilities=caps,
        recommended_next_sources=recommended,
    )


def apply_answer_contract_to_evidence_plan(
    question: str,
    *,
    frame: QuestionFrame,
    plan: EvidencePlan,
    answer_contract: AnswerContract,
    memory_need=None,
) -> EvidencePlan:
    """Turn routed answer scopes into an enforceable evidence plan."""
    normalized_question = " ".join(question.strip().split())
    scopes = set(answer_contract.relevant_scopes)
    memory_query = (
        getattr(memory_need, "query_hint", None)
        or normalized_question
    )

    required_sources: list[str] = []
    discouraged_sources: list[str] = []
    source_queries: dict[str, str] = {}
    source_reasons: dict[str, str] = {}

    if plan.use_memory:
        source_queries["memory"] = memory_query
        source_reasons["memory"] = "Historical continuity or prior discussion is relevant."

    only_historical = scopes == {"historical_discussion"}
    require_artifacts = (
        frame.mode != "remember"
        and bool(scopes & {"repo_current", "install_default"})
    )
    require_runtime = (
        frame.mode != "remember"
        and "runtime_current" in scopes
    )

    if frame.mode == "reconcile" and only_historical:
        plan.use_artifacts = False
        plan.artifact_budget = 0
        plan.use_runtime = False
        plan.runtime_budget = 0

    if require_artifacts:
        plan.use_artifacts = True
        plan.artifact_budget = max(plan.artifact_budget, 5)
        required_sources.append("artifacts")
        source_queries["artifacts"] = _artifact_query_for_question(
            normalized_question,
            answer_contract,
        )
        source_reasons["artifacts"] = (
            "Current repo posture or shipped install defaults are required "
            "for a complete answer."
        )
    elif plan.use_artifacts:
        source_queries["artifacts"] = _artifact_query_for_question(
            normalized_question,
            answer_contract,
        )
        source_reasons["artifacts"] = (
            "Repo artifacts may add supporting context even when they are not required."
        )

    if require_runtime:
        plan.use_runtime = True
        plan.runtime_budget = max(plan.runtime_budget, 1)
        required_sources.append("runtime")
        source_reasons["runtime"] = (
            "Effective runtime state is required to answer the current-scope question."
        )
    elif plan.use_runtime:
        source_reasons["runtime"] = (
            "Runtime state may confirm which mode, profile, or flags are active."
        )

    if frame.mode == "reconcile":
        discouraged_sources.append("facts")
        source_reasons["facts"] = (
            "Generic fact search can expose internal decision-graph edges and "
            "must not replace artifact inspection on reconcile turns."
        )
    elif frame.mode == "remember":
        source_queries["facts"] = normalized_question
        source_reasons["facts"] = (
            "Structured relationship lookup may help when memory recall needs entity facts."
        )

    ordered_recommended = _dedupe(
        required_sources
        + plan.recommended_next_sources
        + (["memory"] if plan.use_memory and "memory" not in required_sources else [])
    )
    plan.recommended_next_sources = ordered_recommended
    plan.required_next_sources = required_sources
    plan.discouraged_sources = _dedupe(discouraged_sources)
    plan.source_queries = source_queries
    plan.source_reasons = source_reasons
    return plan


def infer_claim_state(claim: EvidenceClaim) -> str:
    """Infer how official or settled a normalized claim is."""
    provenance_text = " ".join(
        str(value)
        for value in claim.provenance.values()
        if value is not None
    )
    text = f"{claim.object} {provenance_text}".strip()

    if (
        claim.predicate.upper() == "SUPERSEDED_BY"
        or claim.externalization_state == "superseded"
        or bool(claim.provenance.get("superseded"))
    ):
        return "superseded"
    if claim.source_type == "runtime":
        return "effective"
    if claim.source_type == "artifact":
        path = claim.provenance.get("path", "").lower()
        if claim.externalization_state == "implemented" or path.endswith(
            (".env.example", "pyproject.toml", "package.json")
        ):
            return "implemented"
        return "documented"
    if claim.source_type == "memory":
        if _DECIDED_MARKERS.search(text):
            return "decided"
        if _TENTATIVE_MARKERS.search(text):
            return "tentative"
        if claim.predicate in {"summary", "decision_statement"}:
            return "discussed"
        return "discussed"
    if claim.source_type == "implementation":
        return "implemented"
    return "mentioned"


def apply_claim_states(claims: list[EvidenceClaim]) -> list[EvidenceClaim]:
    """Annotate normalized claims with deterministic claim-state labels."""
    for claim in claims:
        claim.claim_state = infer_claim_state(claim)
    return claims


def summarize_claim_states(claims: list[EvidenceClaim]) -> dict:
    """Build a compact summary for payloads and prompt guidance."""
    if not claims:
        return {
            "counts": {},
            "dominantState": None,
            "bySource": {},
            "byPredicate": {},
        }

    counts = Counter(claim.claim_state for claim in claims)
    by_source: dict[str, Counter] = {}
    by_predicate: dict[str, str] = {}
    rank = {
        "mentioned": 0,
        "discussed": 1,
        "tentative": 2,
        "decided": 3,
        "documented": 4,
        "implemented": 5,
        "effective": 6,
        "superseded": 7,
    }

    for claim in claims:
        by_source.setdefault(claim.source_type, Counter())[claim.claim_state] += 1
        current = by_predicate.get(claim.predicate)
        if current is None or rank.get(claim.claim_state, -1) >= rank.get(current, -1):
            by_predicate[claim.predicate] = claim.claim_state

    dominant_state = max(
        counts.items(),
        key=lambda item: (item[1], rank.get(item[0], -1)),
    )[0]
    return {
        "counts": dict(counts),
        "dominantState": dominant_state,
        "bySource": {source: dict(counter) for source, counter in by_source.items()},
        "byPredicate": by_predicate,
    }


def resolve_answer_contract(
    question: str,
    *,
    frame: QuestionFrame,
    plan: EvidencePlan | None = None,
    claims: list[EvidenceClaim] | None = None,
    reconciliation: ReconciliationResult | None = None,
) -> AnswerContract:
    """Resolve the response operator and scoped answer policy."""
    text = " ".join(question.strip().split())
    lowered = text.lower()
    claims = claims or []
    claim_state_summary = summarize_claim_states(claims)

    plan_markers = bool(_PLAN_PATTERNS.search(text))
    recommend_markers = bool(_RECOMMEND_PATTERNS.search(text))
    compare_markers = bool(_COMPARE_PATTERNS.search(text))
    timeline_markers = bool(_TIMELINE_PATTERNS.search(text))
    settled_markers = bool(_SETTLED_DECISION_PATTERNS.search(text))

    operator = "direct_answer"
    confidence = 0.66
    if plan_markers:
        operator = "plan"
        confidence = 0.88
    elif recommend_markers:
        operator = "recommend"
        confidence = 0.84
    elif compare_markers:
        operator = "compare"
        confidence = 0.9
    elif timeline_markers:
        operator = "timeline"
        confidence = 0.82
    elif frame.mode == "reconcile":
        operator = "reconcile"
        confidence = 0.8

    if claims and settled_markers:
        strong_states = {"decided", "documented", "implemented", "effective"}
        if not any(claim.claim_state in strong_states for claim in claims):
            operator = "unresolved_state_report"
            confidence = max(confidence, 0.86)

    relevant_scopes: list[str] = []
    if frame.mode == "remember":
        relevant_scopes.append("historical_discussion")
    if compare_markers or "default" in lowered:
        relevant_scopes.extend(["raw_default", "install_default"])
    repo_scope_match = bool(_REPO_SCOPE_PATTERNS.search(text))
    if plan and plan.use_artifacts and frame.mode == "inspect":
        relevant_scopes.append("repo_current")
    elif frame.mode == "reconcile" and repo_scope_match:
        relevant_scopes.append("repo_current")
    elif frame.mode == "inspect" and frame.domain in {"project", "product", "mixed"}:
        relevant_scopes.append("repo_current")
    runtime_scope_match = (
        frame.domain == "runtime"
        or compare_markers
        or any(
            token in lowered
            for token in ("runtime", "currently", "right now", "actual", "effective")
        )
    )
    if runtime_scope_match:
        relevant_scopes.append("runtime_current")
    if frame.mode == "reconcile" or "decide" in lowered or "earlier" in lowered:
        relevant_scopes.append("historical_discussion")
    relevant_scopes = _dedupe(relevant_scopes)

    truth_kind = "mixed"
    if frame.mode == "remember" and frame.domain == "personal":
        truth_kind = "personal_continuity"
    elif frame.mode == "reconcile" and "decid" in lowered:
        truth_kind = "historical_intent"
    elif any(scope == "runtime_current" for scope in relevant_scopes) and frame.domain == "runtime":
        truth_kind = "effective_runtime"
    elif "implemented" in lowered or "config" in lowered or "flag" in lowered:
        truth_kind = "implemented_behavior"
    elif frame.mode == "inspect":
        truth_kind = "documented_policy"

    preferred_authorities = list(frame.expected_authorities)
    if operator == "compare":
        preferred_authorities = ["canonical", "current", "historical"]
    elif operator in {"reconcile", "unresolved_state_report", "timeline"}:
        preferred_authorities = ["historical", "canonical", "current"]
    elif operator in {"plan", "recommend"}:
        preferred_authorities = ["canonical", "current", "historical"]

    preserve_temporal_distinction = operator in {
        "compare",
        "reconcile",
        "timeline",
        "unresolved_state_report",
    }
    include_provenance = frame.mode == "reconcile" or operator in {
        "compare",
        "timeline",
        "unresolved_state_report",
    }
    allow_recommendation = operator in {"recommend", "plan"} or (
        operator == "plan" and recommend_markers
    )

    guidance = _answer_contract_guidance(
        operator,
        relevant_scopes=relevant_scopes,
        claim_state_summary=claim_state_summary,
    )

    if reconciliation is not None and reconciliation.status == "conflict":
        preserve_temporal_distinction = True
        include_provenance = True

    return AnswerContract(
        operator=operator,
        requested_truth_kind=truth_kind,
        relevant_scopes=relevant_scopes,
        preferred_authorities=preferred_authorities,
        preserve_temporal_distinction=preserve_temporal_distinction,
        include_provenance=include_provenance,
        allow_recommendation=allow_recommendation,
        confidence=min(0.99, round(confidence, 4)),
        guidance=guidance,
    )


def artifact_class_for_path(rel_path: str) -> str:
    """Map a bootstrapped project file into an artifact class."""
    path = PurePosixPath(rel_path)
    lowered = rel_path.lower()
    name = path.name.lower()
    if name == "readme.md":
        return "readme"
    if name == "skill.md":
        return "skill"
    if name in {".env.example", "pyproject.toml", "package.json", "makefile", "docker-compose.yml"}:
        return "config"
    if lowered.startswith("docs/design/") or lowered.startswith("docs/vision/"):
        return "design_doc"
    if name == "claude.md":
        return "design_doc"
    return "code_file"


def extract_artifact_claims(
    content: str,
    *,
    rel_path: str,
    artifact_class: str,
    project_name: str,
    timestamp: str | None = None,
) -> list[EvidenceClaim]:
    """Extract high-confidence claims from a bootstrapped artifact."""
    claims: list[EvidenceClaim] = []
    seen: set[str] = set()
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    def _append(claim: EvidenceClaim) -> None:
        key = f"{claim.claim_key}::{claim.object.lower()}"
        if key in seen:
            return
        seen.add(key)
        claims.append(claim)

    for line in lines[:80]:
        heading = _HEADING.match(line)
        if heading:
            _append(
                EvidenceClaim(
                    subject=project_name,
                    predicate="heading",
                    object=heading.group(1).strip(),
                    source_type="artifact",
                    authority_type="canonical",
                    externalization_state="documented",
                    timestamp=timestamp,
                    confidence=0.76,
                    provenance={"path": rel_path, "line": line},
                )
            )
            continue

        env_match = _ENV_ASSIGNMENT.match(line)
        if env_match:
            key, value = env_match.groups()
            _append(
                EvidenceClaim(
                    subject=project_name,
                    predicate=f"config:{key.lower()}",
                    object=value.strip().strip('"').strip("'"),
                    source_type="artifact",
                    authority_type="current",
                    externalization_state="implemented",
                    timestamp=timestamp,
                    confidence=0.95,
                    provenance={"path": rel_path, "line": line},
                )
            )
            continue

        json_match = _JSON_STRING.match(line)
        if json_match and json_match.group(1) in {"name", "version"}:
            _append(
                EvidenceClaim(
                    subject=project_name,
                    predicate=f"config:{json_match.group(1)}",
                    object=json_match.group(2),
                    source_type="artifact",
                    authority_type="current",
                    externalization_state="implemented",
                    timestamp=timestamp,
                    confidence=0.82,
                    provenance={"path": rel_path, "line": line},
                )
            )
            continue

        toml_match = _TOML_ASSIGNMENT.match(line)
        if toml_match and toml_match.group(1) in {"name", "version"}:
            _append(
                EvidenceClaim(
                    subject=project_name,
                    predicate=f"config:{toml_match.group(1)}",
                    object=toml_match.group(2).strip(),
                    source_type="artifact",
                    authority_type="current",
                    externalization_state="implemented",
                    timestamp=timestamp,
                    confidence=0.82,
                    provenance={"path": rel_path, "line": line},
                )
            )
            continue

        bullet_match = _BULLET.match(line)
        content_line = bullet_match.group(1).strip() if bullet_match else line
        for predicate, value in _profile_claims_for_text(content_line):
            _append(
                EvidenceClaim(
                    subject=project_name,
                    predicate=predicate,
                    object=value,
                    source_type="artifact",
                    authority_type="current" if artifact_class == "config" else "canonical",
                    externalization_state=(
                        "implemented" if artifact_class == "config" else "documented"
                    ),
                    timestamp=timestamp,
                    confidence=0.92 if artifact_class == "config" else 0.85,
                    provenance={"path": rel_path, "line": content_line},
                )
            )

        for claim in extract_decision_claims(
            content_line,
            subject=project_name,
            source_type="artifact",
            authority_type="canonical" if artifact_class != "config" else "current",
            externalization_state=_artifact_externalization_state(artifact_class, content_line),
            timestamp=timestamp,
            provenance={"path": rel_path, "line": content_line},
        ):
            _append(claim)

    return claims[:16]


def extract_decision_claims(
    text: str,
    *,
    subject: str,
    source_type: str,
    authority_type: str,
    externalization_state: str,
    timestamp: str | None = None,
    provenance: dict | None = None,
) -> list[EvidenceClaim]:
    """Extract decision-like claims from a single line of text."""
    cleaned = " ".join(text.strip().split())
    if not cleaned or not _DECISION_PATTERNS.search(cleaned):
        return []

    claims: list[EvidenceClaim] = []

    if _OPENCLAW_PATTERNS.search(cleaned):
        claims.append(
            EvidenceClaim(
                subject=subject,
                predicate="public_launch_path",
                object="OpenClaw",
                source_type=source_type,
                authority_type=authority_type,
                externalization_state=externalization_state,
                timestamp=timestamp,
                confidence=0.88,
                provenance=dict(provenance or {}),
            )
        )

    if _FULL_MODE_REWORK_PATTERNS.search(cleaned):
        claims.append(
            EvidenceClaim(
                subject=subject,
                predicate="full_mode_default_behavior",
                object="rework",
                source_type=source_type,
                authority_type=authority_type,
                externalization_state=externalization_state,
                timestamp=timestamp,
                confidence=0.82,
                provenance=dict(provenance or {}),
            )
        )

    for predicate, value in _profile_claims_for_text(cleaned):
        claims.append(
            EvidenceClaim(
                subject=subject,
                predicate=predicate,
                object=value,
                source_type=source_type,
                authority_type=authority_type,
                externalization_state=externalization_state,
                timestamp=timestamp,
                confidence=0.84,
                provenance=dict(provenance or {}),
            )
        )

    claims.append(
        EvidenceClaim(
            subject=subject,
            predicate="decision_statement",
            object=cleaned[:240],
            source_type=source_type,
            authority_type=authority_type,
            externalization_state=externalization_state,
            timestamp=timestamp,
            confidence=0.72,
            provenance=dict(provenance or {}),
        )
    )
    return claims


def build_memory_claims(memory_results: list[dict]) -> list[EvidenceClaim]:
    """Convert recall results into normalized memory claims."""
    claims: list[EvidenceClaim] = []
    seen: set[str] = set()
    for result in memory_results:
        if result.get("result_type") != "entity":
            continue
        entity = result.get("entity", {})
        name = entity.get("name") or entity.get("entity") or ""
        summary = (entity.get("summary") or "").strip()
        if name and summary:
            claim = EvidenceClaim(
                subject=name,
                predicate="summary",
                object=summary[:240],
                source_type="memory",
                authority_type="historical",
                externalization_state="discussed",
                confidence=max(0.5, float(result.get("score", 0.0) or 0.0)),
                provenance={"entity_id": entity.get("id")},
            )
            key = f"{claim.claim_key}::{claim.object.lower()}"
            if key not in seen:
                seen.add(key)
                claims.append(claim)
        for rel in result.get("relationships", []):
            subject = rel.get("source_name") or rel.get("subject") or name
            obj = rel.get("target_name") or rel.get("object") or rel.get("target_id")
            predicate = str(rel.get("predicate") or "related_to").lower()
            if not subject or not obj:
                continue
            claim = EvidenceClaim(
                subject=subject,
                predicate=predicate,
                object=str(obj),
                source_type="memory",
                authority_type="historical",
                externalization_state="discussed",
                confidence=max(0.45, float(result.get("score", 0.0) or 0.0)),
                provenance={"entity_id": entity.get("id")},
            )
            key = f"{claim.claim_key}::{claim.object.lower()}"
            if key not in seen:
                seen.add(key)
                claims.append(claim)
    return claims


def build_runtime_claims(runtime_state: dict) -> list[EvidenceClaim]:
    """Convert runtime/config state into normalized current-truth claims."""
    if not runtime_state:
        return []
    subject = runtime_state.get("projectName") or runtime_state.get("name") or "Engram"
    claims: list[EvidenceClaim] = []
    cfg = runtime_state.get("activation", {})
    runtime = runtime_state.get("runtime", {})
    features = runtime_state.get("features", {})

    def _claim(
        predicate: str,
        value: str | bool | int | float,
        confidence: float = 0.96,
    ) -> None:
        claims.append(
            EvidenceClaim(
                subject=subject,
                predicate=predicate,
                object=str(value),
                source_type="runtime",
                authority_type="current",
                externalization_state="effective",
                timestamp=runtime_state.get("generatedAt"),
                confidence=confidence,
                provenance={"source": "runtime_state"},
            )
        )

    if runtime.get("mode") is not None:
        _claim("mode", runtime["mode"])
    if cfg.get("integrationProfile") is not None:
        _claim("integration_profile", cfg["integrationProfile"])
    if cfg.get("recallProfile") is not None:
        _claim("recall_profile", cfg["recallProfile"])
    if cfg.get("consolidationProfile") is not None:
        _claim("consolidation_profile", cfg["consolidationProfile"])
    if features.get("epistemicRoutingEnabled") is not None:
        _claim("epistemic_routing_enabled", features["epistemicRoutingEnabled"], 0.92)
    if features.get("artifactBootstrapEnabled") is not None:
        _claim("artifact_bootstrap_enabled", features["artifactBootstrapEnabled"], 0.92)
    if features.get("answerContractEnabled") is not None:
        _claim("answer_contract_enabled", features["answerContractEnabled"], 0.92)
    if features.get("claimStateModelingEnabled") is not None:
        _claim("claim_state_modeling_enabled", features["claimStateModelingEnabled"], 0.92)
    return claims


def reconcile_claims(
    frame: QuestionFrame,
    *,
    memory_claims: list[EvidenceClaim],
    artifact_claims: list[EvidenceClaim],
    runtime_claims: list[EvidenceClaim],
    implementation_claims: list[EvidenceClaim] | None = None,
    answer_contract: AnswerContract | None = None,
) -> ReconciliationResult:
    """Apply source-aware reconciliation rules to the gathered claims."""
    implementation_claims = implementation_claims or []
    canonical_claims = list(implementation_claims) + list(runtime_claims) + list(artifact_claims)
    supporting = list(memory_claims) + canonical_claims
    sources_used = _sources_from_claims(supporting)
    answer_contract = answer_contract or resolve_answer_contract(
        "",
        frame=frame,
        claims=supporting,
        reconciliation=None,
    )

    if frame.mode == "remember":
        if memory_claims:
            return ReconciliationResult(
                status="confirmed",
                winning_claims=memory_claims[:6],
                supporting_claims=supporting[:12],
                answer_hints=_reconciliation_hints(
                    answer_contract,
                    "Prefer memory continuity and the most recent user framing.",
                ),
                sources_used=sources_used,
            )
        return ReconciliationResult(
            status="insufficient",
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                "Memory did not surface enough supporting evidence.",
            ),
            sources_used=sources_used,
        )

    if frame.mode == "inspect":
        if implementation_claims or runtime_claims:
            winners = (implementation_claims or runtime_claims)[:6]
            return ReconciliationResult(
                status="confirmed",
                winning_claims=winners,
                supporting_claims=supporting[:12],
                answer_hints=_reconciliation_hints(
                    answer_contract,
                    "Prefer current implementation and runtime truth over older discussion.",
                ),
                sources_used=sources_used,
            )
        if artifact_claims:
            return ReconciliationResult(
                status="artifact_only",
                winning_claims=artifact_claims[:6],
                supporting_claims=supporting[:12],
                answer_hints=_reconciliation_hints(
                    answer_contract,
                    (
                        "Answer from repo artifacts and present that as the current "
                        "documented posture."
                    ),
                ),
                sources_used=sources_used,
            )
        if memory_claims:
            return ReconciliationResult(
                status="memory_only",
                winning_claims=memory_claims[:6],
                supporting_claims=supporting[:12],
                answer_hints=_reconciliation_hints(
                    answer_contract,
                    (
                        "Only remembered discussion was found; make clear this is "
                        "not confirmed by artifacts."
                    ),
                ),
                sources_used=sources_used,
            )
        return ReconciliationResult(
            status="insufficient",
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                "Planned inspect sources were exhausted without a clear answer.",
            ),
            sources_used=sources_used,
        )

    agreement, conflicts = _compare_claim_sets(memory_claims, canonical_claims)
    if agreement:
        return ReconciliationResult(
            status="confirmed",
            winning_claims=(agreement + canonical_claims)[:6],
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                (
                    "Memory and current artifacts agree; answer directly with both "
                    "provenance and current posture."
                ),
            ),
            sources_used=sources_used,
        )
    if conflicts:
        return ReconciliationResult(
            status="conflict",
            winning_claims=(canonical_claims[:3] + memory_claims[:3]),
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                (
                    "Preserve the temporal distinction between earlier discussion "
                    "and current documented or implemented state."
                ),
            ),
            sources_used=sources_used,
        )
    if canonical_claims:
        return ReconciliationResult(
            status="artifact_only",
            winning_claims=canonical_claims[:6],
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                (
                    "Current artifacts provide the best available answer even though "
                    "remembered discussion is weak."
                ),
            ),
            sources_used=sources_used,
        )
    if memory_claims:
        return ReconciliationResult(
            status="memory_only",
            winning_claims=memory_claims[:6],
            supporting_claims=supporting[:12],
            answer_hints=_reconciliation_hints(
                answer_contract,
                "This appears to be remembered discussion that is not clearly codified yet.",
            ),
            sources_used=sources_used,
        )
    return ReconciliationResult(
        status="insufficient",
        supporting_claims=supporting[:12],
        answer_hints=_reconciliation_hints(
            answer_contract,
            "All planned sources were exhausted without enough evidence.",
        ),
        sources_used=sources_used,
    )


def render_epistemic_summary(bundle: EpistemicBundle) -> str:
    """Summarize routed evidence for prompts or API responses."""
    lines = [
        (
            "Route: "
            f"{bundle.question_frame.mode} "
            f"({bundle.question_frame.domain}, {bundle.question_frame.timeframe})"
        ),
        (
            "Answer contract: "
            f"{bundle.answer_contract.operator} "
            f"[truth={bundle.answer_contract.requested_truth_kind}]"
        ),
        f"Reconciliation: {bundle.reconciliation.status}",
    ]
    if bundle.answer_contract.relevant_scopes:
        lines.append(
            "Relevant scopes: " + ", ".join(bundle.answer_contract.relevant_scopes[:5])
        )
    if bundle.evidence_plan.required_next_sources:
        lines.append(
            "Required sources: "
            + ", ".join(bundle.evidence_plan.required_next_sources)
        )
    if bundle.evidence_plan.discouraged_sources:
        lines.append(
            "Discouraged sources: "
            + ", ".join(bundle.evidence_plan.discouraged_sources)
        )
    if bundle.evidence_plan.source_queries:
        query_parts = []
        for source in bundle.evidence_plan.required_next_sources or []:
            query = bundle.evidence_plan.source_queries.get(source)
            if query:
                query_parts.append(f"{source}={query}")
        if not query_parts:
            for source in ("memory", "artifacts"):
                query = bundle.evidence_plan.source_queries.get(source)
                if query:
                    query_parts.append(f"{source}={query}")
        if query_parts:
            lines.append("Source queries: " + "; ".join(query_parts[:3]))
    if bundle.claim_state_summary and bundle.claim_state_summary.get("dominantState"):
        lines.append(
            f"Claim-state focus: {bundle.claim_state_summary['dominantState']}"
        )
    if bundle.reconciliation.answer_hints:
        lines.append(f"Policy: {bundle.reconciliation.answer_hints[0]}")
    if bundle.reconciliation.winning_claims:
        lines.append("Winning evidence:")
        for claim in bundle.reconciliation.winning_claims[:4]:
            lines.append(
                f"- [{claim.source_type}/{claim.claim_state}] "
                f"{claim.subject} :: {claim.predicate} -> {claim.object}"
            )
    elif bundle.artifact_hits:
        lines.append("Relevant artifacts:")
        for hit in bundle.artifact_hits[:3]:
            lines.append(f"- {hit.path}: {hit.snippet[:140]}")
    return "\n".join(lines)


def _reconciliation_hints(
    answer_contract: AnswerContract,
    base_hint: str,
) -> list[str]:
    hints = [base_hint]
    for guidance in answer_contract.guidance:
        if guidance not in hints:
            hints.append(guidance)
    return hints


def _answer_contract_guidance(
    operator: str,
    *,
    relevant_scopes: list[str],
    claim_state_summary: dict,
) -> list[str]:
    hints: list[str] = []
    if operator == "compare":
        hints.append("Answer by contrasting scopes instead of flattening to one default.")
    elif operator == "reconcile":
        hints.append("Preserve earlier discussion versus current documented or implemented state.")
    elif operator == "timeline":
        hints.append("Present how the claim moved across states over time.")
    elif operator == "recommend":
        hints.append("State the evidence first, then give advice.")
    elif operator == "plan":
        hints.append("State the current evidence first, then give concrete next steps.")
    elif operator == "unresolved_state_report":
        hints.append("Make clear the issue is not fully settled yet.")
    else:
        hints.append("Answer directly from the highest-authority evidence available.")

    if "runtime_current" in relevant_scopes:
        hints.append("Include the effective runtime state when it materially affects the answer.")
    if "install_default" in relevant_scopes:
        hints.append("Distinguish shipped install defaults from raw code defaults.")
    dominant_state = claim_state_summary.get("dominantState")
    if dominant_state == "tentative":
        hints.append("Treat the strongest evidence as tentative, not final.")
    return hints


def _artifact_query_for_question(
    question: str,
    answer_contract: AnswerContract,
) -> str:
    lowered = question.lower()
    if "openclaw" in lowered and "install" in lowered:
        return "OpenClaw install skill setup"
    if "full mode" in lowered and "rework" in lowered and "default" in lowered:
        return "full mode rework default integration profile"
    if any(token in lowered for token in ("launch", "public", "distribution")):
        return "Engram public launch distribution OpenClaw"
    if "integration_profile" in lowered or "recall_profile" in lowered:
        return question
    if "install_default" in answer_contract.relevant_scopes:
        return f"{question} install default"
    return question


def should_materialize_conversation_decision(text: str) -> bool:
    """Gate conversation decision materialization to strong committed claims."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if cleaned.endswith("?"):
        return False
    if _RECOLLECTION_PROMPTS.search(cleaned):
        return False
    if lowered.startswith(
        (
            "what ",
            "did ",
            "do ",
            "does ",
            "is ",
            "are ",
            "was ",
            "were ",
            "can ",
            "could ",
            "should ",
            "would ",
            "will ",
        )
    ):
        return False
    if _TENTATIVE_MARKERS.search(cleaned) and not _DECIDED_MARKERS.search(cleaned):
        return False
    return bool(_DECIDED_MARKERS.search(cleaned))


def _compare_claim_sets(
    memory_claims: list[EvidenceClaim],
    canonical_claims: list[EvidenceClaim],
) -> tuple[list[EvidenceClaim], list[tuple[EvidenceClaim, EvidenceClaim]]]:
    agreement: list[EvidenceClaim] = []
    conflicts: list[tuple[EvidenceClaim, EvidenceClaim]] = []
    canonical_by_key: dict[str, list[EvidenceClaim]] = {}
    for claim in canonical_claims:
        canonical_by_key.setdefault(claim.claim_key, []).append(claim)
    for memory_claim in memory_claims:
        for canonical_claim in canonical_by_key.get(memory_claim.claim_key, []):
            if _objects_overlap(memory_claim.object, canonical_claim.object):
                agreement.append(canonical_claim)
            else:
                conflicts.append((memory_claim, canonical_claim))
    return agreement, conflicts


def _objects_overlap(left: str, right: str) -> bool:
    left_tokens = {token for token in re.findall(r"\b\w+\b", left.lower()) if len(token) > 2}
    right_tokens = {token for token in re.findall(r"\b\w+\b", right.lower()) if len(token) > 2}
    if not left_tokens or not right_tokens:
        return left.lower() == right.lower()
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return (len(overlap) / max(len(union), 1)) >= 0.35


def _sources_from_claims(claims: list[EvidenceClaim]) -> list[str]:
    return sorted({claim.source_type for claim in claims})


def _artifact_externalization_state(artifact_class: str, line: str) -> str:
    lowered = line.lower()
    if artifact_class == "config":
        return "implemented"
    if "launch" in lowered or "public" in lowered or "distribution" in lowered:
        return "announced"
    return "documented"


def _profile_claims_for_text(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    claims: list[tuple[str, str]] = []
    for pattern, predicate, value in _PROFILE_PATTERNS:
        if pattern.search(text):
            claims.append((predicate, value))
    if "full mode" in lowered and "rework" in lowered and "default" in lowered:
        claims.append(("full_mode_default_behavior", "rework"))
    return claims


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
