"""Models for epistemic routing, evidence planning, and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionFrame:
    """Deterministic framing of what kind of truth the user is asking for."""

    mode: str
    domain: str
    timeframe: str
    expected_authorities: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    requires_workspace: bool = False
    confidence: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "domain": self.domain,
            "timeframe": self.timeframe,
            "expectedAuthorities": self.expected_authorities,
            "expectedSources": self.expected_sources,
            "requiresWorkspace": self.requires_workspace,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass
class EvidencePlan:
    """Bounded executor plan for the framed question."""

    use_memory: bool = False
    use_artifacts: bool = False
    use_implementation: bool = False
    use_runtime: bool = False
    memory_budget: int = 0
    artifact_budget: int = 0
    implementation_budget: int = 0
    runtime_budget: int = 0
    surface_capabilities: dict[str, bool] = field(default_factory=dict)
    recommended_next_sources: list[str] = field(default_factory=list)
    required_next_sources: list[str] = field(default_factory=list)
    discouraged_sources: list[str] = field(default_factory=list)
    source_queries: dict[str, str] = field(default_factory=dict)
    source_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "useMemory": self.use_memory,
            "useArtifacts": self.use_artifacts,
            "useImplementation": self.use_implementation,
            "useRuntime": self.use_runtime,
            "budgets": {
                "memory": self.memory_budget,
                "artifacts": self.artifact_budget,
                "implementation": self.implementation_budget,
                "runtime": self.runtime_budget,
            },
            "surfaceCapabilities": self.surface_capabilities,
            "recommendedNextSources": self.recommended_next_sources,
            "requiredNextSources": self.required_next_sources,
            "discouragedSources": self.discouraged_sources,
            "sourceQueries": self.source_queries,
            "sourceReasons": self.source_reasons,
        }


@dataclass
class EvidenceClaim:
    """Normalized claim extracted from memory, artifacts, or runtime state."""

    subject: str
    predicate: str
    object: str
    source_type: str
    authority_type: str
    externalization_state: str
    claim_state: str = "mentioned"
    timestamp: str | None = None
    confidence: float = 0.0
    provenance: dict = field(default_factory=dict)

    @property
    def claim_key(self) -> str:
        return f"{self.subject.lower()}::{self.predicate.lower()}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "sourceType": self.source_type,
            "authorityType": self.authority_type,
            "externalizationState": self.externalization_state,
            "claimState": self.claim_state,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance,
        }


@dataclass
class ArtifactHit:
    """Search result from the artifact substrate."""

    artifact_id: str
    path: str
    artifact_class: str
    snippet: str
    last_observed_at: str | None = None
    score: float = 0.0
    stale: bool = False
    supporting_claims: list[EvidenceClaim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "artifactId": self.artifact_id,
            "path": self.path,
            "artifactClass": self.artifact_class,
            "snippet": self.snippet,
            "lastObservedAt": self.last_observed_at,
            "score": round(self.score, 4),
            "stale": self.stale,
            "supportingClaims": [claim.to_dict() for claim in self.supporting_claims],
        }


@dataclass
class ReconciliationResult:
    """Result of reconciling claims across sources."""

    status: str
    winning_claims: list[EvidenceClaim] = field(default_factory=list)
    supporting_claims: list[EvidenceClaim] = field(default_factory=list)
    answer_hints: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "winningClaims": [claim.to_dict() for claim in self.winning_claims],
            "supportingClaims": [claim.to_dict() for claim in self.supporting_claims],
            "answerHints": self.answer_hints,
            "sourcesUsed": self.sources_used,
        }


@dataclass
class AnswerContract:
    """Response-shaping policy derived from question intent and evidence state."""

    operator: str
    requested_truth_kind: str
    relevant_scopes: list[str] = field(default_factory=list)
    preferred_authorities: list[str] = field(default_factory=list)
    preserve_temporal_distinction: bool = False
    include_provenance: bool = False
    allow_recommendation: bool = False
    confidence: float = 0.0
    guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "requestedTruthKind": self.requested_truth_kind,
            "relevantScopes": self.relevant_scopes,
            "preferredAuthorities": self.preferred_authorities,
            "preserveTemporalDistinction": self.preserve_temporal_distinction,
            "includeProvenance": self.include_provenance,
            "allowRecommendation": self.allow_recommendation,
            "confidence": round(self.confidence, 4),
            "guidance": self.guidance,
        }


@dataclass
class EpistemicBundle:
    """Full routed evidence package for internal orchestration."""

    question_frame: QuestionFrame
    evidence_plan: EvidencePlan
    reconciliation: ReconciliationResult
    answer_contract: AnswerContract
    memory_claims: list[EvidenceClaim] = field(default_factory=list)
    artifact_claims: list[EvidenceClaim] = field(default_factory=list)
    runtime_claims: list[EvidenceClaim] = field(default_factory=list)
    implementation_claims: list[EvidenceClaim] = field(default_factory=list)
    artifact_hits: list[ArtifactHit] = field(default_factory=list)
    memory_results: list[dict] = field(default_factory=list)
    runtime_state: dict | None = None
    claim_state_summary: dict | None = None

    def to_dict(self) -> dict:
        return {
            "questionFrame": self.question_frame.to_dict(),
            "evidencePlan": self.evidence_plan.to_dict(),
            "reconciliation": self.reconciliation.to_dict(),
            "answerContract": self.answer_contract.to_dict(),
            "memoryClaims": [claim.to_dict() for claim in self.memory_claims],
            "artifactClaims": [claim.to_dict() for claim in self.artifact_claims],
            "runtimeClaims": [claim.to_dict() for claim in self.runtime_claims],
            "implementationClaims": [claim.to_dict() for claim in self.implementation_claims],
            "artifactHits": [hit.to_dict() for hit in self.artifact_hits],
            "memoryResults": self.memory_results,
            "runtimeState": self.runtime_state,
            "claimStateSummary": self.claim_state_summary,
        }
