"""Data models for the showcase benchmark suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


TurnAction = Literal["observe", "remember", "project", "intend", "dismiss_intention"]
ProbeOperation = Literal["recall", "get_context"]
TrackName = Literal["showcase", "answer", "external", "all"]


def estimate_tokens(text: str) -> int:
    """Approximate token count using the repo's existing 4 chars/token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-safe primitives."""
    if is_dataclass(value):
        return {
            key: to_serializable(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


@dataclass(frozen=True)
class BudgetProfile:
    """Shared retrieval + answer budget contract for a scenario."""

    retrieval_limit: int = 5
    evidence_max_tokens: int = 220
    answer_budget_tokens: int = 120


@dataclass(frozen=True)
class ExtractionSpec:
    """Deterministic extraction fixture for a single turn."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerTask:
    """Optional answer-level evaluation metadata for a scenario."""

    question: str
    gold_answer: Any
    answer_grader: str = "field_match"
    expected_format: str = "json"
    answer_budget_tokens: int = 120
    note: str = ""


@dataclass(frozen=True)
class ScenarioTurn:
    """A single action applied to a baseline adapter."""

    id: str
    action: TurnAction
    content: str | None = None
    source: str = "showcase"
    session_id: str | None = None
    ref: str | None = None
    extraction: ExtractionSpec | None = None
    trigger_text: str | None = None
    action_text: str | None = None
    trigger_type: str = "activation"
    entity_names: list[str] = field(default_factory=list)
    threshold: float | None = None
    priority: str = "normal"
    context: str | None = None
    see_also: list[str] = field(default_factory=list)
    hard_delete: bool = False


@dataclass(frozen=True)
class ScenarioProbe:
    """A deterministic probe evaluated after a specific turn index."""

    id: str
    after_turn_index: int
    operation: ProbeOperation
    query: str | None = None
    topic_hint: str | None = None
    limit: int = 5
    max_tokens: int = 220
    required_evidence: list[str] = field(default_factory=list)
    required_evidence_result_types: list[str] = field(default_factory=list)
    forbidden_evidence: list[str] = field(default_factory=list)
    expected_result_types: list[str] = field(default_factory=list)
    allowed_result_types: list[str] = field(default_factory=list)
    disallowed_result_types: list[str] = field(default_factory=list)
    historical_evidence_allowed: bool = True
    capability_tags: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class ShowcaseScenario:
    """End-to-end deterministic scenario for the showcase benchmark."""

    id: str
    title: str
    why_it_matters: str
    turns: list[ScenarioTurn]
    probes: list[ScenarioProbe]
    capability_tags: list[str] = field(default_factory=list)
    answer_task: AnswerTask | None = None
    gold_answer: Any = None
    answer_grader: str | None = None
    budget_profile: BudgetProfile = field(default_factory=BudgetProfile)
    distractor_tags: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """Single surfaced evidence item returned by a baseline."""

    result_type: str
    text: str
    source_id: str | None = None
    score: float = 0.0
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            self.tokens = estimate_tokens(self.text)


@dataclass
class AdapterCostStats:
    """Cost and behavior counters for a single adapter run."""

    observed_turns: int = 0
    projected_turns: int = 0
    extraction_calls: int = 0
    embedding_calls: int = 0
    consolidation_cycles: int = 0
    method_calls: dict[str, int] = field(default_factory=dict)

    @property
    def selective_extraction_ratio(self) -> float:
        if self.observed_turns <= 0:
            return 0.0
        return self.projected_turns / self.observed_turns

    def bump(self, method_name: str, amount: int = 1) -> None:
        self.method_calls[method_name] = self.method_calls.get(method_name, 0) + amount


@dataclass
class ProbeResult:
    """Scored result for a single probe."""

    probe_id: str
    passed: bool
    required_hits: list[str]
    missing_required: list[str]
    forbidden_hits: list[str]
    expected_type_match: bool
    returned_types: list[str]
    latency_ms: float
    tokens_surfaced: int
    required_hit_rate: float = 0.0
    forbidden_hit_rate: float = 0.0
    token_efficiency: float = 0.0
    disallowed_type_hits: list[str] = field(default_factory=list)
    historical_violation: bool = False
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Showcase-track scenario result for one baseline/seed pair."""

    scenario_id: str
    scenario_title: str
    why_it_matters: str
    baseline_name: str
    baseline_family: str
    seed: int
    capability_tags: list[str]
    available: bool
    passed: bool
    probe_results: list[ProbeResult] = field(default_factory=list)
    cost_stats: AdapterCostStats = field(default_factory=AdapterCostStats)
    availability_reason: str | None = None


@dataclass
class BaselineSummary:
    """Aggregated showcase-track metrics for a baseline."""

    baseline_name: str
    baseline_family: str
    is_ablation: bool
    available: bool
    availability_reason: str | None
    scenario_pass_rate: float
    capability_pass_rates: dict[str, float]
    false_recall_rate: float
    temporal_correctness: float
    negation_correctness: float
    open_loop_recovery: float
    prospective_trigger_rate: float
    required_hit_rate: float
    forbidden_hit_rate: float
    token_efficiency: float
    tokens_per_passed_scenario: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_proxies: dict[str, float]


@dataclass
class AnswerResult:
    """Answer-track result for one scenario/baseline/seed pair."""

    scenario_id: str
    scenario_title: str
    baseline_name: str
    baseline_family: str
    seed: int
    available: bool
    passed: bool
    answer_task_question: str
    answer: Any = None
    normalized_answer: Any = None
    score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    incorrect_fields: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_surfaced: int = 0
    availability_reason: str | None = None


@dataclass
class AnswerSummary:
    """Aggregated answer-track metrics for a baseline."""

    baseline_name: str
    baseline_family: str
    available: bool
    availability_reason: str | None
    answer_pass_rate: float
    average_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    tokens_per_passed_answer: float


@dataclass
class ExternalTrackResult:
    """Status/summary for an external supporting benchmark track."""

    name: str
    available: bool
    executed: bool
    availability_reason: str | None = None
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    recommended_command: str | None = None


@dataclass
class FairnessContract:
    """Frozen benchmark contract reported with every run."""

    track: str
    strict_fairness: bool
    scenario_budgets: dict[str, dict[str, int]]
    vector_provider_family: str
    answer_model: str | None
    answer_provider: str | None
    answer_prompt: str | None
    transcript_invariant: bool
    baseline_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    transcript_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class TrackSummary:
    """Availability/execution summary for one benchmark track."""

    track: str
    executed: bool
    available: bool
    availability_reason: str | None = None
    headline_metric: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ShowcaseRunResult:
    """Full suite output suitable for JSON and report generation."""

    track: str
    mode: str
    seeds: list[int]
    generated_at: str
    output_dir: str
    fairness_contract: FairnessContract
    primary_baselines: list[str]
    appendix_baselines: list[str]
    ablation_baselines: list[str]
    scenario_results: list[ScenarioResult]
    baseline_summaries: list[BaselineSummary]
    answer_results: list[AnswerResult] = field(default_factory=list)
    answer_summaries: list[AnswerSummary] = field(default_factory=list)
    external_track_results: list[ExternalTrackResult] = field(default_factory=list)
    track_summaries: list[TrackSummary] = field(default_factory=list)
    supporting_artifacts: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    readme_snippet: str | None = None
