import type {
  GraphNode,
  GraphEdge,
  AtlasData,
  AtlasHistoryEntry,
  RegionData,
  SearchResult,
  EntityDetail,
  Episode,
  GraphStats,
  RuntimeState,
  BrainLoopEvaluationReport,
  HumanLabelEvidence,
  AdoptionEvidence,
  AdoptionClientEvidence,
  AdoptionClientEvidenceReport,
  LifecycleSummary,
  GraphRepresentationMeta,
  ConsolidationCycleSummary,
  ConsolidationCycleDetail,
  ConsolidationPressure,
  RecallResponse,
  FactResult,
  IntentionItem,
  RecallEvaluationInput,
  RecallEvaluationWriteResponse,
  SessionContinuityEvaluationInput,
  SessionContinuityEvaluationWriteResponse,
} from "../store/types";

export interface NeighborhoodResponse {
  centerId: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  representation?: GraphRepresentationMeta;
  truncated: boolean;
  totalInNeighborhood: number;
}

export interface AtlasHistoryResponse {
  items: AtlasHistoryEntry[];
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  sessionDate: string;
  createdAt: string;
  updatedAt: string;
  entityIds: string[];
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  partsJson: string | null;
  createdAt: string;
}

export interface EpisodesResponse {
  items: Episode[];
  nextCursor: string | null;
}

export type ServerHealthStatus = "healthy" | "degraded" | "unhealthy";

export interface HealthResponse {
  status: ServerHealthStatus;
  version: string;
  mode: string;
  services: Record<string, ServerHealthStatus>;
}

interface RawStatsResponse {
  stats?: {
    entities?: number;
    relationships?: number;
    episodes?: number;
    entity_type_distribution?: Record<string, number>;
    cue_metrics?: {
      cue_count?: number;
      episodes_without_cues?: number;
      cue_coverage?: number;
      cue_hit_count?: number;
      cue_hit_episode_count?: number;
      cue_hit_episode_rate?: number;
      cue_surfaced_count?: number;
      cue_selected_count?: number;
      cue_used_count?: number;
      cue_near_miss_count?: number;
      avg_policy_score?: number;
      avg_projection_attempts?: number;
      projected_cue_count?: number;
      cue_to_projection_conversion_rate?: number;
    };
    projection_metrics?: {
      state_counts?: {
        queued?: number;
        cued?: number;
        cue_only?: number;
        scheduled?: number;
        projecting?: number;
        projected?: number;
        failed?: number;
        dead_letter?: number;
      };
      attempted_episode_count?: number;
      total_attempts?: number;
      failure_count?: number;
      dead_letter_count?: number;
      failure_rate?: number;
      avg_processing_duration_ms?: number;
      avg_time_to_projection_ms?: number;
      yield?: {
        linked_entity_count?: number;
        relationship_count?: number;
        avg_linked_entities_per_projected_episode?: number;
        avg_relationships_per_projected_episode?: number;
      };
    };
    adjudication_metrics?: {
      evidence_status_counts?: {
        pending?: number;
        deferred?: number;
        approved?: number;
      };
      request_status_counts?: {
        pending?: number;
        deferred?: number;
        error?: number;
      };
      open_evidence_count?: number;
      pending_evidence_count?: number;
      deferred_evidence_count?: number;
      approved_evidence_count?: number;
      open_request_count?: number;
      pending_request_count?: number;
      deferred_request_count?: number;
      error_request_count?: number;
      open_work_count?: number;
    };
  };
  topActivated?: Array<{
    id: string;
    name: string;
    entityType: string;
    activationCurrent?: number;
    activation?: number;
  }>;
  topConnected?: Array<{
    id: string;
    name: string;
    entityType: string;
    edgeCount?: number;
    connectionCount?: number;
  }>;
  growthTimeline?: GraphStats["growthTimeline"];
}

interface RawEvaluationReport {
  group_id?: string;
  generated_at?: string;
  loop?: BrainLoopEvaluationReport["loop"];
  totals?: {
    episodes?: number;
    entities?: number;
    relationships?: number;
    active_entities?: number;
  };
  capture?: {
    status?: string;
    episode_count?: number;
    active_count?: number;
  };
  cue?: {
    status?: string;
    cue_count?: number;
    episodes_without_cues?: number;
    coverage?: number;
    hit_count?: number;
    hit_episode_count?: number;
    hit_episode_rate?: number;
    surfaced_count?: number;
    selected_count?: number;
    used_count?: number;
    near_miss_count?: number;
    selected_rate?: number;
    used_rate?: number;
    near_miss_rate?: number;
    avg_policy_score?: number;
    projection_conversion_rate?: number;
  };
  project?: {
    status?: string;
    state_counts?: {
      queued?: number;
      cued?: number;
      cue_only?: number;
      scheduled?: number;
      projecting?: number;
      projected?: number;
      merged?: number;
      failed?: number;
      dead_letter?: number;
    };
    tracked_count?: number;
    projected_count?: number;
    active_count?: number;
    projected_rate?: number;
    backlog_rate?: number;
    failed_count?: number;
    dead_letter_count?: number;
    attempted_episode_count?: number;
    total_attempts?: number;
    failure_rate?: number;
    avg_processing_duration_ms?: number;
    avg_time_to_projection_ms?: number;
    yield?: {
      linked_entity_count?: number;
      relationship_count?: number;
      avg_linked_entities_per_projected_episode?: number;
      avg_relationships_per_projected_episode?: number;
    };
  };
  recall?: {
    status?: string;
    total_analyses?: number;
    trigger_count?: number;
    runtime_false_recall_rate?: number;
    runtime_surfaced_to_used_ratio?: number | null;
    graph_lift_rate?: number;
    probe_trigger_rate?: number;
    latency?: {
      analyzer_ms?: RawLatencySummary;
      analyzerMs?: RawLatencySummary;
      probe_ms?: RawLatencySummary;
      probeMs?: RawLatencySummary;
    };
    control?: {
      used_count?: number;
      usedCount?: number;
      dismissed_count?: number;
      dismissedCount?: number;
      surfaced_count?: number;
      surfacedCount?: number;
      selected_count?: number;
      selectedCount?: number;
      confirmed_count?: number;
      confirmedCount?: number;
      corrected_count?: number;
      correctedCount?: number;
      graph_override_count?: number;
      graphOverrideCount?: number;
      adaptive_thresholds_enabled?: boolean;
      adaptiveThresholdsEnabled?: boolean;
      thresholds?: {
        linguistic?: number;
        borderline?: number;
        resonance?: number;
      };
    };
    family_contributions?: Record<string, number>;
    evaluation?: {
      status?: string;
      sample_count?: number;
      need_status?: string;
      need_labeled_count?: number;
      needed_count?: number;
      missed_count?: number;
      memory_need_precision?: number | null;
      memory_need_recall?: number | null;
      missed_recall_rate?: number | null;
      useful_packet_rate?: number | null;
      false_recall_rate?: number | null;
      surfaced_count?: number;
      used_count?: number;
      surfaced_to_used_ratio?: number | null;
    };
    continuity?: {
      status?: string;
      sample_count?: number;
      session_continuity_lift?: number | null;
      open_loop_recovery_rate?: number | null;
      temporal_correctness?: number | null;
    };
  };
  consolidate?: {
    status?: string;
    cycle_count?: number;
    latest_status?: string | null;
    latest_cycle?: Record<string, unknown> | null;
    phase_status_counts?: Record<string, number>;
    phase_totals?: Record<
      string,
      {
        runs?: number;
        items_processed?: number;
        items_affected?: number;
        effect_rate?: number;
      }
    >;
    adjudication?: {
      status?: string;
      phase_count?: number;
      runs?: number;
      items_processed?: number;
      items_affected?: number;
      items_unaffected?: number;
      effect_rate?: number;
      error_count?: number;
      open_evidence_count?: number;
      open_request_count?: number;
      open_work_count?: number;
      pending_evidence_count?: number;
      deferred_evidence_count?: number;
      approved_evidence_count?: number;
      pending_request_count?: number;
      deferred_request_count?: number;
      error_request_count?: number;
      evidence_status_counts?: Record<string, number>;
      request_status_counts?: Record<string, number>;
      phase_totals?: Record<
        string,
        {
          runs?: number;
          items_processed?: number;
          items_affected?: number;
          effect_rate?: number;
        }
      >;
    };
    calibration?: {
      status?: string;
      snapshot_count?: number;
      phase_totals?: Record<
        string,
        {
          snapshots?: number;
          total_traces?: number;
          labeled_examples?: number;
          oracle_examples?: number;
          abstain_count?: number;
          accuracy?: number | null;
          mean_confidence?: number | null;
          expected_calibration_error?: number | null;
        }
      >;
    };
    items_processed?: number;
    items_affected?: number;
    effect_rate?: number;
    error_count?: number;
  };
  evaluation_signals?: RawEvaluationSignals;
  evaluationSignals?: RawEvaluationSignals;
  coverage_gaps?: string[];
  coverageGaps?: string[];
  release_evidence?: unknown;
  releaseEvidence?: unknown;
  human_label_evidence?: unknown;
  humanLabelEvidence?: unknown;
  adoption_evidence?: unknown;
  adoptionEvidence?: unknown;
  additional_adoption_evidence?: unknown;
  additionalAdoptionEvidence?: unknown;
  adoption_client_evidence?: unknown;
  adoptionClientEvidence?: unknown;
}

interface RawLatencySummary {
  avg?: number;
  p95?: number;
  avg_ms?: number;
  avgMs?: number;
  p95_ms?: number;
  p95Ms?: number;
}

type RawEvaluationSignal = {
  status?: string;
  evidence_count?: number;
  evidenceCount?: number;
  metric?: number | null;
  gap?: string | null;
};

type RawEvaluationSignals = Record<string, RawEvaluationSignal | undefined>;

type RawPhaseTotals = NonNullable<NonNullable<RawEvaluationReport["consolidate"]>["phase_totals"]>;
type RawAdjudicationPhaseTotals = NonNullable<
  NonNullable<NonNullable<RawEvaluationReport["consolidate"]>["adjudication"]>["phase_totals"]
>;
type RawCalibrationPhaseTotals = NonNullable<
  NonNullable<NonNullable<RawEvaluationReport["consolidate"]>["calibration"]>["phase_totals"]
>;

const API_BASE = import.meta.env.VITE_API_URL ?? "";

// Auth token getter — set by auth provider (e.g. Clerk)
let _getToken: (() => Promise<string | null>) | null = null;

export function setAuthTokenGetter(fn: () => Promise<string | null>) {
  _getToken = fn;
}

export async function getAuthToken(): Promise<string | null> {
  return _getToken ? _getToken() : null;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const fullUrl = url.startsWith("/") ? `${API_BASE}${url}` : url;
  const headers = new Headers(init?.headers);
  if (_getToken) {
    const token = await _getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(fullUrl, { ...init, headers });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

function mapPhaseTotals(totals?: RawPhaseTotals) {
  return Object.fromEntries(
    Object.entries(totals ?? {}).map(([phase, value]) => [
      phase,
      {
        runs: value.runs ?? 0,
        itemsProcessed: value.items_processed ?? 0,
        itemsAffected: value.items_affected ?? 0,
        effectRate: value.effect_rate ?? 0,
      },
    ]),
  );
}

function mapLatencySummary(summary?: RawLatencySummary) {
  return {
    avgMs: summary?.avg_ms ?? summary?.avgMs ?? summary?.avg ?? 0,
    p95Ms: summary?.p95_ms ?? summary?.p95Ms ?? summary?.p95 ?? 0,
  };
}

function mapEvaluationSignal(raw: RawEvaluationSignal | undefined) {
  return {
    status: raw?.status ?? "needs_data",
    evidenceCount: raw?.evidence_count ?? raw?.evidenceCount ?? 0,
    metric: raw?.metric ?? null,
    gap: raw?.gap ?? null,
  };
}

function mapEvaluationSignals(
  raw?: RawEvaluationSignals,
): BrainLoopEvaluationReport["evaluationSignals"] {
  return {
    cueUsefulness: mapEvaluationSignal(raw?.cue_usefulness ?? raw?.cueUsefulness),
    projectionYield: mapEvaluationSignal(raw?.projection_yield ?? raw?.projectionYield),
    recallQuality: mapEvaluationSignal(raw?.recall_quality ?? raw?.recallQuality),
    falseRecall: mapEvaluationSignal(raw?.false_recall ?? raw?.falseRecall),
    triageCalibration: mapEvaluationSignal(
      raw?.triage_calibration ?? raw?.triageCalibration,
    ),
    consolidationEffect: mapEvaluationSignal(
      raw?.consolidation_effect ?? raw?.consolidationEffect,
    ),
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readField(record: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(record, key)) return record[key];
  }
  return undefined;
}

function stringOrNull(value: unknown): string | null {
  if (typeof value === "string") return value || null;
  if (value == null) return null;
  return String(value);
}

function numberOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function booleanOrFalse(value: unknown): boolean {
  return value === true;
}

function booleanOrNull(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  return null;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => stringOrNull(item))
    .filter((item): item is string => item != null);
}

function evidenceStatus(raw: Record<string, unknown> | null) {
  return stringOrNull(raw ? readField(raw, "status") : null) ?? "missing";
}

function mapHumanLabelEvidence(raw: unknown): HumanLabelEvidence | null {
  const data = asRecord(raw);
  if (!data) return null;
  return {
    status: evidenceStatus(data),
    artifactPath: stringOrNull(readField(data, "artifact_path", "artifactPath")),
    artifactSha256: stringOrNull(readField(data, "artifact_sha256", "artifactSha256")),
    kind: stringOrNull(readField(data, "kind")),
    source: stringOrNull(readField(data, "source")),
    client: stringOrNull(readField(data, "client")),
    capturedAt: stringOrNull(readField(data, "captured_at", "capturedAt")),
    sessionId: stringOrNull(readField(data, "session_id", "sessionId")),
    labeler: stringOrNull(readField(data, "labeler")),
    humanLabeled: booleanOrFalse(readField(data, "human_labeled", "humanLabeled")),
    recallSampleCount: numberOrZero(readField(data, "recall_sample_count", "recallSampleCount")),
    sessionSampleCount: numberOrZero(readField(data, "session_sample_count", "sessionSampleCount")),
    minRecallSamples: numberOrZero(readField(data, "min_recall_samples", "minRecallSamples")),
    minSessionSamples: numberOrZero(readField(data, "min_session_samples", "minSessionSamples")),
    sampleSources: stringList(readField(data, "sample_sources", "sampleSources")),
    failures: stringList(readField(data, "failures")),
  };
}

function mapAdoptionEvidence(raw: unknown): AdoptionEvidence | null {
  const data = asRecord(raw);
  if (!data) return null;
  const requiredTools = asRecord(readField(data, "required_tools", "requiredTools")) ?? {};
  const capture = asRecord(readField(data, "capture")) ?? {};
  const fileMemory = asRecord(readField(data, "file_memory", "fileMemory")) ?? {};
  return {
    status: evidenceStatus(data),
    artifactPath: stringOrNull(readField(data, "artifact_path", "artifactPath")),
    artifactSha256: stringOrNull(readField(data, "artifact_sha256", "artifactSha256")),
    adoptionStatus: stringOrNull(readField(data, "adoption_status", "adoptionStatus")),
    authorityPath: stringOrNull(readField(data, "authority_path", "authorityPath")),
    callsPath: stringOrNull(readField(data, "calls_path", "callsPath")),
    callCount: numberOrZero(readField(data, "call_count", "callCount")),
    client: stringOrNull(readField(data, "client")),
    requiredClient: stringOrNull(readField(data, "required_client", "requiredClient")),
    gateRequiredClient: stringOrNull(readField(data, "gate_required_client", "gateRequiredClient")),
    capturedAt: stringOrNull(readField(data, "captured_at", "capturedAt")),
    sessionId: stringOrNull(readField(data, "session_id", "sessionId")),
    sessionFilter: stringOrNull(readField(data, "session_filter", "sessionFilter")),
    source: stringOrNull(readField(data, "source")),
    requiredLiveEvidence: booleanOrFalse(
      readField(data, "required_live_evidence", "requiredLiveEvidence"),
    ),
    blockers: stringList(readField(data, "blockers")),
    blockerDetails: stringList(readField(data, "blocker_details", "blockerDetails")),
    mcpServerFailures: stringList(
      readField(data, "mcp_server_failures", "mcpServerFailures"),
    ),
    requiredTools: {
      expected: stringList(readField(requiredTools, "expected")),
      observed: stringList(readField(requiredTools, "observed")),
      missing: stringList(readField(requiredTools, "missing")),
      inOrder: booleanOrFalse(readField(requiredTools, "in_order", "inOrder")),
    },
    capture: {
      destination: stringOrNull(readField(capture, "destination")),
      expectedTool: stringOrNull(readField(capture, "expected_tool", "expectedTool")),
      observedTools: stringList(readField(capture, "observed_tools", "observedTools")),
      missing: booleanOrFalse(readField(capture, "missing")),
    },
    fileMemory: {
      present: booleanOrNull(readField(fileMemory, "present")),
      substitutedForEngram: booleanOrFalse(
        readField(fileMemory, "substituted_for_engram", "substitutedForEngram"),
      ),
    },
    failures: stringList(readField(data, "failures")),
  };
}

function mapAdoptionClientEvidenceReport(raw: unknown): AdoptionClientEvidenceReport | null {
  const data = asRecord(raw);
  if (!data) return null;
  return {
    client: stringOrNull(readField(data, "client")),
    requiredClient: stringOrNull(readField(data, "required_client", "requiredClient")),
    status: evidenceStatus(data),
    artifactPath: stringOrNull(readField(data, "artifact_path", "artifactPath")),
    artifactSha256: stringOrNull(readField(data, "artifact_sha256", "artifactSha256")),
    capturedAt: stringOrNull(readField(data, "captured_at", "capturedAt")),
    sessionId: stringOrNull(readField(data, "session_id", "sessionId")),
    blockers: stringList(readField(data, "blockers")),
    blockerDetails: stringList(readField(data, "blocker_details", "blockerDetails")),
    mcpServerFailures: stringList(
      readField(data, "mcp_server_failures", "mcpServerFailures"),
    ),
    failures: stringList(readField(data, "failures")),
  };
}

function mapAdoptionClientEvidence(raw: unknown): AdoptionClientEvidence | null {
  const data = asRecord(raw);
  if (!data) return null;
  const reports = (Array.isArray(readField(data, "reports"))
    ? (readField(data, "reports") as unknown[])
    : []
  )
    .map(mapAdoptionClientEvidenceReport)
    .filter((item): item is AdoptionClientEvidenceReport => item != null);
  return {
    status: evidenceStatus(data),
    requiredClients: stringList(readField(data, "required_clients", "requiredClients")),
    observedClients: stringList(readField(data, "observed_clients", "observedClients")),
    reportCount: numberOrZero(readField(data, "report_count", "reportCount")) || reports.length,
    reports,
    blockers: stringList(readField(data, "blockers")),
    mcpServerFailures: stringList(
      readField(data, "mcp_server_failures", "mcpServerFailures"),
    ),
    failures: stringList(readField(data, "failures")),
  };
}

function mapReleaseEvidenceComponent(raw: unknown) {
  const data = asRecord(raw);
  return {
    status: evidenceStatus(data),
    missing: stringList(data ? readField(data, "missing") : null),
    failures: stringList(data ? readField(data, "failures") : null),
    requiredClients: stringList(
      data ? readField(data, "required_clients", "requiredClients") : null,
    ),
    observedClients: stringList(
      data ? readField(data, "observed_clients", "observedClients") : null,
    ),
    blockers: stringList(data ? readField(data, "blockers") : null),
    blockerDetails: stringList(
      data ? readField(data, "blocker_details", "blockerDetails") : null,
    ),
    mcpServerFailures: stringList(
      data ? readField(data, "mcp_server_failures", "mcpServerFailures") : null,
    ),
  };
}

function mapReleaseEvidence(raw: unknown): BrainLoopEvaluationReport["releaseEvidence"] {
  const data = asRecord(raw);
  if (!data) return null;
  const components = asRecord(readField(data, "components")) ?? {};
  return {
    status: evidenceStatus(data),
    components: {
      evaluationSignals: mapReleaseEvidenceComponent(
        readField(components, "evaluation_signals", "evaluationSignals"),
      ),
      humanLabels: mapReleaseEvidenceComponent(
        readField(components, "human_labels", "humanLabels"),
      ),
      adoption: mapReleaseEvidenceComponent(readField(components, "adoption")),
      adoptionClients: mapReleaseEvidenceComponent(
        readField(components, "adoption_clients", "adoptionClients"),
      ),
    },
    missing: stringList(readField(data, "missing")),
    failures: stringList(readField(data, "failures")),
  };
}

function mapAdditionalAdoptionEvidence(raw: unknown): AdoptionEvidence[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map(mapAdoptionEvidence)
    .filter((item): item is AdoptionEvidence => item != null);
}

function mapCalibrationPhaseTotals(totals?: RawCalibrationPhaseTotals) {
  return Object.fromEntries(
    Object.entries(totals ?? {}).map(([phase, value]) => [
      phase,
      {
        snapshots: value.snapshots ?? 0,
        totalTraces: value.total_traces ?? 0,
        labeledExamples: value.labeled_examples ?? 0,
        oracleExamples: value.oracle_examples ?? 0,
        abstainCount: value.abstain_count ?? 0,
        accuracy: value.accuracy ?? null,
        meanConfidence: value.mean_confidence ?? null,
        expectedCalibrationError: value.expected_calibration_error ?? null,
      },
    ]),
  );
}

export const api = {
  getHealth: () => fetchJSON<HealthResponse>("/health"),

  getRuntimeState: (params?: { projectPath?: string | null }) => {
    const sp = new URLSearchParams();
    if (params?.projectPath) sp.set("project_path", params.projectPath);
    const suffix = sp.size > 0 ? `?${sp}` : "";
    return fetchJSON<RuntimeState>(`/api/knowledge/runtime${suffix}`);
  },

  getGraphAtlas: (params?: { refresh?: boolean; snapshotId?: string | null }) => {
    const sp = new URLSearchParams();
    if (params?.refresh) sp.set("refresh", "true");
    if (params?.snapshotId) sp.set("snapshot_id", params.snapshotId);
    const suffix = sp.size > 0 ? `?${sp}` : "";
    return fetchJSON<AtlasData>(`/api/graph/atlas${suffix}`);
  },

  getGraphAtlasHistory: (params?: { limit?: number }) => {
    const sp = new URLSearchParams();
    if (params?.limit != null) sp.set("limit", String(params.limit));
    const suffix = sp.size > 0 ? `?${sp}` : "";
    return fetchJSON<AtlasHistoryResponse>(`/api/graph/atlas/history${suffix}`);
  },

  getGraphRegion: (
    regionId: string,
    params?: { refresh?: boolean; snapshotId?: string | null },
  ) => {
    const sp = new URLSearchParams();
    if (params?.refresh) sp.set("refresh", "true");
    if (params?.snapshotId) sp.set("snapshot_id", params.snapshotId);
    const suffix = sp.size > 0 ? `?${sp}` : "";
    return fetchJSON<RegionData>(`/api/graph/regions/${regionId}${suffix}`);
  },

  getNeighborhood: (params: {
    center?: string;
    depth?: number;
    maxNodes?: number;
    minActivation?: number;
  }) => {
    const sp = new URLSearchParams();
    if (params.center) sp.set("center", params.center);
    if (params.depth) sp.set("depth", String(params.depth));
    if (params.maxNodes != null) sp.set("max_nodes", String(params.maxNodes));
    if (params.minActivation)
      sp.set("min_activation", String(params.minActivation));
    return fetchJSON<NeighborhoodResponse>(
      `/api/graph/neighborhood?${sp}`,
    );
  },

  searchEntities: async (
    params: { q?: string; type?: string; limit?: number; signal?: AbortSignal },
  ): Promise<SearchResult[]> => {
    const sp = new URLSearchParams();
    if (params.q) sp.set("q", params.q);
    if (params.type) sp.set("type", params.type);
    if (params.limit) sp.set("limit", String(params.limit));
    const data = await fetchJSON<{
      items: Array<{
        id: string;
        name: string;
        entityType: string;
        summary: string | null;
        lexicalRegime?: string | null;
        canonicalIdentifier?: string | null;
        identifierLabel?: boolean;
        activationCurrent: number;
      }>;
      total: number;
    }>(
      `/api/entities/search?${sp}`,
      { signal: params.signal },
    );
    return data.items.map((item) => ({
      id: item.id,
      name: item.name,
      entityType: item.entityType,
      summary: item.summary,
      lexicalRegime: item.lexicalRegime ?? null,
      canonicalIdentifier: item.canonicalIdentifier ?? null,
      identifierLabel: Boolean(item.identifierLabel),
      activationScore: item.activationCurrent,
    }));
  },

  getEntity: (id: string) => fetchJSON<EntityDetail>(`/api/entities/${id}`),

  getNeighbors: (
    id: string,
    params?: { depth?: number; maxNodes?: number },
  ) => {
    const sp = new URLSearchParams();
    if (params?.depth) sp.set("depth", String(params.depth));
    if (params?.maxNodes) sp.set("max_nodes", String(params.maxNodes));
    return fetchJSON<NeighborhoodResponse>(
      `/api/entities/${id}/neighbors?${sp}`,
    );
  },

  getStats: async (): Promise<GraphStats> => {
    const raw = await fetchJSON<RawStatsResponse>("/api/stats");
    const s = raw.stats ?? {};
    const cueMetrics = s.cue_metrics;
    const projectionMetrics = s.projection_metrics;
    const projectionStateCounts = projectionMetrics?.state_counts;
    const projectionYield = projectionMetrics?.yield;
    const adjudicationMetrics = s.adjudication_metrics;
    const evidenceStatusCounts = adjudicationMetrics?.evidence_status_counts;
    const requestStatusCounts = adjudicationMetrics?.request_status_counts;
    return {
      totalEntities: s.entities ?? 0,
      totalRelationships: s.relationships ?? 0,
      totalEpisodes: s.episodes ?? 0,
      entityTypeCounts: s.entity_type_distribution ?? {},
      cueMetrics: {
        cueCount: cueMetrics?.cue_count ?? 0,
        episodesWithoutCues: cueMetrics?.episodes_without_cues ?? 0,
        cueCoverage: cueMetrics?.cue_coverage ?? 0,
        cueHitCount: cueMetrics?.cue_hit_count ?? 0,
        cueHitEpisodeCount: cueMetrics?.cue_hit_episode_count ?? 0,
        cueHitEpisodeRate: cueMetrics?.cue_hit_episode_rate ?? 0,
        cueSurfacedCount: cueMetrics?.cue_surfaced_count ?? 0,
        cueSelectedCount: cueMetrics?.cue_selected_count ?? 0,
        cueUsedCount: cueMetrics?.cue_used_count ?? 0,
        cueNearMissCount: cueMetrics?.cue_near_miss_count ?? 0,
        avgPolicyScore: cueMetrics?.avg_policy_score ?? 0,
        avgProjectionAttempts: cueMetrics?.avg_projection_attempts ?? 0,
        projectedCueCount: cueMetrics?.projected_cue_count ?? 0,
        cueToProjectionConversionRate:
          cueMetrics?.cue_to_projection_conversion_rate ?? 0,
      },
      projectionMetrics: {
        stateCounts: {
          queued: projectionStateCounts?.queued ?? 0,
          cued: projectionStateCounts?.cued ?? 0,
          cueOnly: projectionStateCounts?.cue_only ?? 0,
          scheduled: projectionStateCounts?.scheduled ?? 0,
          projecting: projectionStateCounts?.projecting ?? 0,
          projected: projectionStateCounts?.projected ?? 0,
          failed: projectionStateCounts?.failed ?? 0,
          deadLetter: projectionStateCounts?.dead_letter ?? 0,
        },
        attemptedEpisodeCount: projectionMetrics?.attempted_episode_count ?? 0,
        totalAttempts: projectionMetrics?.total_attempts ?? 0,
        failureCount: projectionMetrics?.failure_count ?? 0,
        deadLetterCount: projectionMetrics?.dead_letter_count ?? 0,
        failureRate: projectionMetrics?.failure_rate ?? 0,
        avgProcessingDurationMs:
          projectionMetrics?.avg_processing_duration_ms ?? 0,
        avgTimeToProjectionMs:
          projectionMetrics?.avg_time_to_projection_ms ?? 0,
        yield: {
          linkedEntityCount: projectionYield?.linked_entity_count ?? 0,
          relationshipCount: projectionYield?.relationship_count ?? 0,
          avgLinkedEntitiesPerProjectedEpisode:
            projectionYield?.avg_linked_entities_per_projected_episode ?? 0,
          avgRelationshipsPerProjectedEpisode:
            projectionYield?.avg_relationships_per_projected_episode ?? 0,
        },
      },
      adjudicationMetrics: {
        evidenceStatusCounts: {
          pending: evidenceStatusCounts?.pending ?? 0,
          deferred: evidenceStatusCounts?.deferred ?? 0,
          approved: evidenceStatusCounts?.approved ?? 0,
        },
        requestStatusCounts: {
          pending: requestStatusCounts?.pending ?? 0,
          deferred: requestStatusCounts?.deferred ?? 0,
          error: requestStatusCounts?.error ?? 0,
        },
        openEvidenceCount: adjudicationMetrics?.open_evidence_count ?? 0,
        pendingEvidenceCount: adjudicationMetrics?.pending_evidence_count ?? 0,
        deferredEvidenceCount: adjudicationMetrics?.deferred_evidence_count ?? 0,
        approvedEvidenceCount: adjudicationMetrics?.approved_evidence_count ?? 0,
        openRequestCount: adjudicationMetrics?.open_request_count ?? 0,
        pendingRequestCount: adjudicationMetrics?.pending_request_count ?? 0,
        deferredRequestCount: adjudicationMetrics?.deferred_request_count ?? 0,
        errorRequestCount: adjudicationMetrics?.error_request_count ?? 0,
        openWorkCount: adjudicationMetrics?.open_work_count ?? 0,
      },
      topActivated: (raw.topActivated ?? []).map((a) => ({
        id: a.id,
        name: a.name,
        entityType: a.entityType,
        activation: a.activationCurrent ?? a.activation ?? 0,
      })),
      topConnected: (raw.topConnected ?? []).map((c) => ({
        id: c.id,
        name: c.name,
        entityType: c.entityType,
        connectionCount: c.edgeCount ?? c.connectionCount ?? 0,
      })),
      growthTimeline: raw.growthTimeline ?? [],
    };
  },

  getLifecycleSummary: () =>
    fetchJSON<LifecycleSummary>("/api/lifecycle/summary"),

  getEvaluationReport: async (): Promise<BrainLoopEvaluationReport> => {
    const raw = await fetchJSON<RawEvaluationReport>("/api/evaluation/brain-loop/report");
    const totals = raw.totals ?? {};
    const capture = raw.capture ?? {};
    const cue = raw.cue ?? {};
    const project = raw.project ?? {};
    const stateCounts = project.state_counts ?? {};
    const projectYield = project.yield ?? {};
    const recall = raw.recall ?? {};
    const evaluation = recall.evaluation ?? {};
      const control = recall.control ?? {};
      const thresholds = control.thresholds ?? {};
      const continuity = recall.continuity ?? {};
      const consolidate = raw.consolidate ?? {};
      const adjudication = consolidate.adjudication ?? {};
      const calibration = consolidate.calibration ?? {};

    return {
      groupId: raw.group_id ?? "default",
      generatedAt: raw.generated_at ?? "",
      loop: raw.loop ?? ["capture", "cue", "project", "recall", "consolidate"],
      totals: {
        episodes: totals.episodes ?? 0,
        entities: totals.entities ?? 0,
        relationships: totals.relationships ?? 0,
        activeEntities: totals.active_entities ?? 0,
      },
      capture: {
        status: capture.status ?? "ready",
        episodeCount: capture.episode_count ?? 0,
        activeCount: capture.active_count ?? 0,
      },
      cue: {
        status: cue.status ?? "ready",
        cueCount: cue.cue_count ?? 0,
        episodesWithoutCues: cue.episodes_without_cues ?? 0,
        coverage: cue.coverage ?? 0,
        hitCount: cue.hit_count ?? 0,
        hitEpisodeCount: cue.hit_episode_count ?? 0,
        hitEpisodeRate: cue.hit_episode_rate ?? 0,
        surfacedCount: cue.surfaced_count ?? 0,
        selectedCount: cue.selected_count ?? 0,
        usedCount: cue.used_count ?? 0,
        nearMissCount: cue.near_miss_count ?? 0,
        selectedRate: cue.selected_rate ?? 0,
        usedRate: cue.used_rate ?? 0,
        nearMissRate: cue.near_miss_rate ?? 0,
        avgPolicyScore: cue.avg_policy_score ?? 0,
        projectionConversionRate: cue.projection_conversion_rate ?? 0,
      },
      project: {
        status: project.status ?? "ready",
        stateCounts: {
          queued: stateCounts.queued ?? 0,
          cued: stateCounts.cued ?? 0,
          cueOnly: stateCounts.cue_only ?? 0,
          scheduled: stateCounts.scheduled ?? 0,
          projecting: stateCounts.projecting ?? 0,
          projected: stateCounts.projected ?? 0,
          merged: stateCounts.merged ?? 0,
          failed: stateCounts.failed ?? 0,
          deadLetter: stateCounts.dead_letter ?? 0,
        },
        trackedCount: project.tracked_count ?? 0,
        projectedCount: project.projected_count ?? 0,
        activeCount: project.active_count ?? 0,
        projectedRate: project.projected_rate ?? 0,
        backlogRate: project.backlog_rate ?? 0,
        failedCount: project.failed_count ?? 0,
        deadLetterCount: project.dead_letter_count ?? 0,
        attemptedEpisodeCount: project.attempted_episode_count ?? 0,
        totalAttempts: project.total_attempts ?? 0,
        failureRate: project.failure_rate ?? 0,
        avgProcessingDurationMs: project.avg_processing_duration_ms ?? 0,
        avgTimeToProjectionMs: project.avg_time_to_projection_ms ?? 0,
        yield: {
          linkedEntityCount: projectYield.linked_entity_count ?? 0,
          relationshipCount: projectYield.relationship_count ?? 0,
          avgLinkedEntitiesPerProjectedEpisode:
            projectYield.avg_linked_entities_per_projected_episode ?? 0,
          avgRelationshipsPerProjectedEpisode:
            projectYield.avg_relationships_per_projected_episode ?? 0,
        },
      },
      recall: {
        status: recall.status ?? "ready",
        totalAnalyses: recall.total_analyses ?? 0,
        triggerCount: recall.trigger_count ?? 0,
        runtimeFalseRecallRate: recall.runtime_false_recall_rate ?? 0,
        runtimeSurfacedToUsedRatio: recall.runtime_surfaced_to_used_ratio ?? null,
        graphLiftRate: recall.graph_lift_rate ?? 0,
        probeTriggerRate: recall.probe_trigger_rate ?? 0,
        latency: {
          analyzerMs: mapLatencySummary(
            recall.latency?.analyzer_ms ?? recall.latency?.analyzerMs,
          ),
          probeMs: mapLatencySummary(recall.latency?.probe_ms ?? recall.latency?.probeMs),
        },
        control: {
          usedCount: control.used_count ?? control.usedCount ?? 0,
          dismissedCount: control.dismissed_count ?? control.dismissedCount ?? 0,
          surfacedCount: control.surfaced_count ?? control.surfacedCount ?? 0,
          selectedCount: control.selected_count ?? control.selectedCount ?? 0,
          confirmedCount: control.confirmed_count ?? control.confirmedCount ?? 0,
          correctedCount: control.corrected_count ?? control.correctedCount ?? 0,
          graphOverrideCount:
            control.graph_override_count ?? control.graphOverrideCount ?? 0,
          adaptiveThresholdsEnabled:
            control.adaptive_thresholds_enabled ?? control.adaptiveThresholdsEnabled ?? false,
          thresholds: {
            linguistic: thresholds.linguistic ?? 0,
            borderline: thresholds.borderline ?? 0,
            resonance: thresholds.resonance ?? 0,
          },
        },
        familyContributions: recall.family_contributions ?? {},
        evaluation: {
          status: evaluation.status ?? "needs_samples",
          sampleCount: evaluation.sample_count ?? 0,
          needStatus: evaluation.need_status ?? "needs_samples",
          needLabeledCount: evaluation.need_labeled_count ?? 0,
          neededCount: evaluation.needed_count ?? 0,
          missedCount: evaluation.missed_count ?? 0,
          memoryNeedPrecision: evaluation.memory_need_precision ?? null,
          memoryNeedRecall: evaluation.memory_need_recall ?? null,
          missedRecallRate: evaluation.missed_recall_rate ?? null,
          usefulPacketRate: evaluation.useful_packet_rate ?? null,
          falseRecallRate: evaluation.false_recall_rate ?? null,
          surfacedCount: evaluation.surfaced_count ?? 0,
          usedCount: evaluation.used_count ?? 0,
          surfacedToUsedRatio: evaluation.surfaced_to_used_ratio ?? null,
        },
        continuity: {
          status: continuity.status ?? "needs_samples",
          sampleCount: continuity.sample_count ?? 0,
          sessionContinuityLift: continuity.session_continuity_lift ?? null,
          openLoopRecoveryRate: continuity.open_loop_recovery_rate ?? null,
          temporalCorrectness: continuity.temporal_correctness ?? null,
        },
      },
      consolidate: {
        status: consolidate.status ?? "needs_cycles",
        cycleCount: consolidate.cycle_count ?? 0,
        latestStatus: consolidate.latest_status ?? null,
        latestCycle: consolidate.latest_cycle ?? null,
        phaseStatusCounts: consolidate.phase_status_counts ?? {},
        phaseTotals: mapPhaseTotals(consolidate.phase_totals),
        adjudication: {
          status: adjudication.status ?? "needs_cycles",
          phaseCount: adjudication.phase_count ?? 0,
          runs: adjudication.runs ?? 0,
          itemsProcessed: adjudication.items_processed ?? 0,
          itemsAffected: adjudication.items_affected ?? 0,
          itemsUnaffected: adjudication.items_unaffected ?? 0,
          effectRate: adjudication.effect_rate ?? 0,
          errorCount: adjudication.error_count ?? 0,
          openEvidenceCount: adjudication.open_evidence_count ?? 0,
          openRequestCount: adjudication.open_request_count ?? 0,
          openWorkCount: adjudication.open_work_count ?? 0,
          pendingEvidenceCount: adjudication.pending_evidence_count ?? 0,
          deferredEvidenceCount: adjudication.deferred_evidence_count ?? 0,
          approvedEvidenceCount: adjudication.approved_evidence_count ?? 0,
          pendingRequestCount: adjudication.pending_request_count ?? 0,
          deferredRequestCount: adjudication.deferred_request_count ?? 0,
          errorRequestCount: adjudication.error_request_count ?? 0,
          evidenceStatusCounts: adjudication.evidence_status_counts ?? {},
          requestStatusCounts: adjudication.request_status_counts ?? {},
          phaseTotals: mapPhaseTotals(adjudication.phase_totals as RawAdjudicationPhaseTotals),
        },
        calibration: {
          status: calibration.status ?? "needs_snapshots",
          snapshotCount: calibration.snapshot_count ?? 0,
          phaseTotals: mapCalibrationPhaseTotals(calibration.phase_totals),
        },
        itemsProcessed: consolidate.items_processed ?? 0,
        itemsAffected: consolidate.items_affected ?? 0,
        effectRate: consolidate.effect_rate ?? 0,
        errorCount: consolidate.error_count ?? 0,
      },
      evaluationSignals: mapEvaluationSignals(
        raw.evaluation_signals ?? raw.evaluationSignals,
      ),
      coverageGaps: raw.coverage_gaps ?? raw.coverageGaps ?? [],
      releaseEvidence: mapReleaseEvidence(raw.release_evidence ?? raw.releaseEvidence),
      humanLabelEvidence: mapHumanLabelEvidence(
        raw.human_label_evidence ?? raw.humanLabelEvidence,
      ),
      adoptionEvidence: mapAdoptionEvidence(
        raw.adoption_evidence ?? raw.adoptionEvidence,
      ),
      additionalAdoptionEvidence: mapAdditionalAdoptionEvidence(
        raw.additional_adoption_evidence ?? raw.additionalAdoptionEvidence,
      ),
      adoptionClientEvidence: mapAdoptionClientEvidence(
        raw.adoption_client_evidence ?? raw.adoptionClientEvidence,
      ),
    };
  },

  recordRecallEvaluation: (params: RecallEvaluationInput) =>
    fetchJSON<RecallEvaluationWriteResponse>("/api/evaluation/recall-samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  recordSessionContinuityEvaluation: (params: SessionContinuityEvaluationInput) =>
    fetchJSON<SessionContinuityEvaluationWriteResponse>("/api/evaluation/session-samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  getEpisodes: (params?: { cursor?: string; limit?: number; source?: string; status?: string }) => {
    const sp = new URLSearchParams();
    if (params?.cursor) sp.set("cursor", params.cursor);
    if (params?.limit) sp.set("limit", String(params.limit));
    if (params?.source) sp.set("source", params.source);
    if (params?.status) sp.set("status", params.status);
    return fetchJSON<EpisodesResponse>(`/api/episodes?${sp}`);
  },

  getGraphAt: (timestamp: string, centerId?: string) => {
    const sp = new URLSearchParams();
    sp.set("at", timestamp);
    if (centerId) sp.set("center", centerId);
    return fetchJSON<NeighborhoodResponse>(`/api/graph/at?${sp}`);
  },

  updateEntity: (id: string, body: { name?: string; summary?: string }) =>
    fetchJSON<Record<string, unknown>>(`/api/entities/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  deleteEntity: (id: string) =>
    fetchJSON<Record<string, unknown>>(`/api/entities/${id}`, {
      method: "DELETE",
    }),

  getActivationSnapshot: (limit?: number) => {
    const sp = new URLSearchParams();
    if (limit) sp.set("limit", String(limit));
    return fetchJSON<{ topActivated: Array<{ entityId: string; name: string; entityType: string; currentActivation: number; accessCount: number; lastAccessedAt: string | null; decayRate: number }> }>(`/api/activation/snapshot?${sp}`);
  },

  getActivationCurve: (entityId: string, hours?: number, points?: number) => {
    const sp = new URLSearchParams();
    if (hours) sp.set("hours", String(hours));
    if (points) sp.set("points", String(points));
    return fetchJSON<{ entityId: string; entityName: string; curve: Array<{ timestamp: string; activation: number }>; accessEvents: string[]; formula: string; hours: number; points: number }>(`/api/activation/${entityId}/curve?${sp}`);
  },

  getConsolidationStatus: () =>
    fetchJSON<{ is_running: boolean; scheduler_active: boolean; pressure?: ConsolidationPressure; latest_cycle?: ConsolidationCycleSummary }>("/api/consolidation/status"),

  getConsolidationHistory: (limit = 10) =>
    fetchJSON<{ cycles: ConsolidationCycleSummary[] }>(`/api/consolidation/history?limit=${limit}`),

  getConsolidationCycle: (cycleId: string) =>
    fetchJSON<ConsolidationCycleDetail>(`/api/consolidation/cycle/${cycleId}`),

  triggerConsolidation: (dryRun = true) =>
    fetchJSON<{ status: "triggered"; group_id: string; dry_run: boolean }>("/api/consolidation/trigger?dry_run=" + dryRun, { method: "POST" }),

  // Knowledge API
  recall: (params: { q: string; limit?: number }) => {
    const sp = new URLSearchParams();
    sp.set("q", params.q);
    if (params.limit) sp.set("limit", String(params.limit));
    return fetchJSON<RecallResponse>(`/api/knowledge/recall?${sp}`);
  },

  searchFacts: (params: { q: string; subject?: string; predicate?: string; limit?: number }) => {
    const sp = new URLSearchParams();
    sp.set("q", params.q);
    if (params.subject) sp.set("subject", params.subject);
    if (params.predicate) sp.set("predicate", params.predicate);
    if (params.limit) sp.set("limit", String(params.limit));
    return fetchJSON<{ items: FactResult[] }>(`/api/knowledge/facts?${sp}`);
  },

  getKnowledgeContext: (params?: { maxTokens?: number; topicHint?: string }) => {
    const sp = new URLSearchParams();
    if (params?.maxTokens) sp.set("max_tokens", String(params.maxTokens));
    if (params?.topicHint) sp.set("topic_hint", params.topicHint);
    return fetchJSON<{ context: string; entityCount: number; factCount: number; tokenEstimate: number }>(`/api/knowledge/context?${sp}`);
  },

  observe: (params: { content: string; source?: string }) =>
    fetchJSON<{ status: string; episodeId: string }>("/api/knowledge/observe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  remember: (params: { content: string; source?: string }) =>
    fetchJSON<{ status: string; episodeId: string }>("/api/knowledge/remember", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  forget: (params: { entity_name?: string; fact?: { subject: string; predicate: string; object: string }; reason?: string }) =>
    fetchJSON<{ status: string }>("/api/knowledge/forget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  getIntentions: (enabledOnly = true) => {
    const sp = new URLSearchParams();
    sp.set("enabled_only", String(enabledOnly));
    return fetchJSON<{ intentions: IntentionItem[]; total: number }>(`/api/knowledge/intentions?${sp}`);
  },

  // Conversations API
  listConversations: (limit = 50) =>
    fetchJSON<{ conversations: ConversationSummary[] }>(`/api/conversations/?limit=${limit}`),

  getConversationMessages: (convId: string, init?: RequestInit) =>
    fetchJSON<{ messages: ConversationMessage[] }>(
      `/api/conversations/${convId}/messages`,
      init,
    ),

  appendConversationMessages: (convId: string, messages: Array<{ role: string; content: string; partsJson?: string }>) =>
    fetchJSON<{ ids: string[] }>(`/api/conversations/${convId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    }),

  updateConversation: (convId: string, body: { title?: string }) =>
    fetchJSON<{ status: string }>(`/api/conversations/${convId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  deleteConversation: (convId: string) =>
    fetchJSON<{ status: string }>(`/api/conversations/${convId}`, {
      method: "DELETE",
    }),

  submitFeedback: (params: { entityId: string; rating: number; comment?: string }) =>
    fetchJSON<{ status: string; entity_id: string; domain: string; edge_type: string | null; edge_weight: number }>("/api/knowledge/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entity_id: params.entityId, rating: params.rating, comment: params.comment }),
    }),
};
