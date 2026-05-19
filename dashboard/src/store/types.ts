export interface GraphNode {
  id: string;
  name: string;
  entityType: string;
  summary: string | null;
  activationCurrent: number;
  accessCount: number;
  lastAccessed: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  predicate: string;
  weight: number;
  validFrom: string | null;
  validTo: string | null;
  createdAt: string;
}

export type BrainMapScope = "atlas" | "region" | "neighborhood" | "temporal";

export interface GraphRepresentationMeta {
  scope: BrainMapScope;
  layout: "precomputed" | "force";
  representedEntityCount: number;
  representedEdgeCount: number;
  displayedNodeCount: number;
  displayedEdgeCount: number;
  snapshotId?: string;
  truncated: boolean;
}

export interface AtlasRegion {
  id: string;
  label: string;
  subtitle: string | null;
  kind: "identity" | "mixed" | "domain" | "schema_cluster";
  memberCount: number;
  representedEdgeCount: number;
  activationScore: number;
  growth7d: number;
  growth30d: number;
  dominantEntityTypes: Record<string, number>;
  hubEntityIds: string[];
  centerEntityId: string | null;
  latestEntityCreatedAt: string | null;
  x: number;
  y: number;
  z: number;
}

export interface AtlasBridge {
  id: string;
  source: string;
  target: string;
  weight: number;
  relationshipCount: number;
}

export interface AtlasData {
  representation: GraphRepresentationMeta;
  generatedAt: string;
  regions: AtlasRegion[];
  bridges: AtlasBridge[];
  stats: {
    totalEntities: number;
    totalRelationships: number;
    totalRegions: number;
    hottestRegionId: string | null;
    fastestGrowingRegionId: string | null;
  };
}

export interface AtlasHistoryEntry {
  id: string;
  generatedAt: string;
  representedEntityCount: number;
  representedEdgeCount: number;
  displayedNodeCount: number;
  displayedEdgeCount: number;
  totalEntities: number;
  totalRelationships: number;
  totalRegions: number;
  hottestRegionId: string | null;
  fastestGrowingRegionId: string | null;
  truncated: boolean;
}

export interface RegionGraphNode {
  id: string;
  kind: "hub" | "schema" | "cluster" | "bridge";
  label: string;
  representedEntityCount: number;
  activationScore: number;
  growth30d: number;
  x: number;
  y: number;
  z: number;
  entityId?: string;
  entityType?: string;
  regionId?: string;
}

export interface RegionGraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  predicateHint: string | null;
}

export interface RegionData {
  representation: GraphRepresentationMeta;
  generatedAt: string;
  region: {
    id: string;
    label: string;
    subtitle: string | null;
    kind: "identity" | "mixed" | "domain" | "schema_cluster";
    memberCount: number;
    activationScore: number;
    growth7d: number;
    growth30d: number;
    latestEntityCreatedAt: string | null;
  };
  nodes: RegionGraphNode[];
  edges: RegionGraphEdge[];
  topEntities: Array<{
    id: string;
    name: string;
    entityType: string;
    activationCurrent: number;
  }>;
  memberIds: string[];
}

export interface SearchResult {
  id: string;
  name: string;
  entityType: string;
  summary: string | null;
  lexicalRegime?: string | null;
  canonicalIdentifier?: string | null;
  identifierLabel?: boolean;
  activationScore: number;
}

export interface EntityFact {
  id: string;
  predicate: string;
  direction: "outgoing" | "incoming";
  other: { id: string; name: string; entityType: string };
  weight: number;
  validFrom: string | null;
  validTo: string | null;
  createdAt: string;
}

export interface EntityDetail {
  id: string;
  name: string;
  entityType: string;
  summary: string | null;
  lexicalRegime?: string | null;
  canonicalIdentifier?: string | null;
  identifierLabel?: boolean;
  activationCurrent: number;
  accessCount: number;
  lastAccessed: string | null;
  createdAt: string;
  updatedAt: string;
  facts: EntityFact[];
}

export type EpisodeStatus =
  | "queued"
  | "extracting"
  | "resolving"
  | "writing"
  | "embedding"
  | "activating"
  | "completed"
  | "retrying"
  | "dead_letter"
  | "pending"
  | "processing"
  | "failed";

export type EpisodeProjectionState =
  | "queued"
  | "cued"
  | "cue_only"
  | "scheduled"
  | "projecting"
  | "projected"
  | "merged"
  | "failed"
  | "dead_letter";

export interface EpisodeCueSummary {
  cueText: string | null;
  projectionState: EpisodeProjectionState | null;
  routeReason: string | null;
  hitCount: number;
  surfacedCount: number;
  selectedCount: number;
  usedCount: number;
  nearMissCount: number;
  policyScore: number | null;
  projectionAttempts: number;
  lastHitAt: string | null;
  lastFeedbackAt: string | null;
  lastProjectedAt: string | null;
}

export interface Episode {
  episodeId: string;
  content: string;
  source: string;
  status: EpisodeStatus;
  projectionState?: EpisodeProjectionState | null;
  lastProjectionReason?: string | null;
  lastProjectedAt?: string | null;
  conversationDate?: string | null;
  createdAt: string;
  updatedAt: string;
  entities: Array<{ id: string; name: string; entityType: string }>;
  factsCount: number;
  processingDurationMs: number | null;
  error: string | null;
  retryCount: number;
  cue?: EpisodeCueSummary | null;
}

export interface GraphStats {
  totalEntities: number;
  totalRelationships: number;
  totalEpisodes: number;
  entityTypeCounts: Record<string, number>;
  cueMetrics?: {
    cueCount: number;
    episodesWithoutCues: number;
    cueCoverage: number;
    cueHitCount: number;
    cueHitEpisodeCount: number;
    cueHitEpisodeRate: number;
    cueSurfacedCount: number;
    cueSelectedCount: number;
    cueUsedCount: number;
    cueNearMissCount: number;
    avgPolicyScore: number;
    avgProjectionAttempts: number;
    projectedCueCount: number;
    cueToProjectionConversionRate: number;
  };
  projectionMetrics?: {
    stateCounts: {
      queued: number;
      cued: number;
      cueOnly: number;
      scheduled: number;
      projecting: number;
      projected: number;
      failed: number;
      deadLetter: number;
    };
    attemptedEpisodeCount: number;
    totalAttempts: number;
    failureCount: number;
    deadLetterCount: number;
    failureRate: number;
    avgProcessingDurationMs: number;
    avgTimeToProjectionMs: number;
    yield: {
      linkedEntityCount: number;
      relationshipCount: number;
      avgLinkedEntitiesPerProjectedEpisode: number;
      avgRelationshipsPerProjectedEpisode: number;
    };
  };
  adjudicationMetrics?: {
    evidenceStatusCounts: {
      pending: number;
      deferred: number;
      approved: number;
    };
    requestStatusCounts: {
      pending: number;
      deferred: number;
      error: number;
    };
    openEvidenceCount: number;
    pendingEvidenceCount: number;
    deferredEvidenceCount: number;
    approvedEvidenceCount: number;
    openRequestCount: number;
    pendingRequestCount: number;
    deferredRequestCount: number;
    errorRequestCount: number;
    openWorkCount: number;
  };
  topActivated: Array<{ id: string; name: string; entityType: string; activation: number }>;
  topConnected: Array<{ id: string; name: string; entityType: string; connectionCount: number }>;
  growthTimeline: Array<{ date: string; entities: number; episodes: number }>;
}

export type AgentAdoptionStatus =
  | "fresh_runtime"
  | "needs_project_bootstrap"
  | "ready";

export interface RuntimeState {
  projectName: string;
  runtime: {
    mode: string;
  };
  activation: Record<string, string | number | boolean | null | undefined>;
  features: Record<string, boolean>;
  artifactBootstrap: {
    enabled: boolean;
    projectPath: string | null;
    artifactCount: number;
    freshArtifactCount: number;
    staleArtifactCount: number;
    lastObservedAt: string | null;
    staleAfterSeconds: number;
  };
  agentAdoption: {
    status: AgentAdoptionStatus;
    doNotTreatEmptyAsFailure: boolean;
    requiredNextTools: string[];
    beforeAnswer?: {
      required: boolean;
      tools: string[];
      reason: string;
    };
    emptyRuntimePolicy?: string;
    fileMemoryPolicy?: string;
    claimAuthority: {
      tool: "claim_authority";
      args: Record<string, string | boolean | null>;
      reason: string;
    };
    bootstrap: {
      tool: "bootstrap_project";
      required: boolean;
      args: { project_path: string };
      reason: string;
    };
    reason: string;
  };
  stats: {
    recallMetrics: Record<string, unknown>;
    epistemicMetrics: Record<string, unknown>;
  };
  generatedAt: string;
}

export type LifecycleStageKey = "capture" | "cue" | "project" | "recall" | "consolidate";
export type LifecycleStageStatus = "active" | "ready" | "attention";

export interface LifecycleSummary {
  groupId: string;
  generatedAt: string;
  loop: LifecycleStageKey[];
  totals: {
    episodes: number;
    cues: number;
    projected: number;
    cycles: number;
    entities: number;
    relationships: number;
  };
  capture: {
    status: LifecycleStageStatus;
    episodeCount: number;
    activeCount: number;
    latestEpisode: Episode | null;
  };
  cue: {
    status: LifecycleStageStatus;
    cueCount: number;
    episodesWithoutCues: number;
    coverage: number;
    hitCount: number;
    surfacedCount: number;
    selectedCount: number;
    usedCount: number;
    nearMissCount: number;
    avgPolicyScore: number;
    projectionConversionRate: number;
  };
  project: {
    status: LifecycleStageStatus;
    projectedCount: number;
    activeCount: number;
    failedCount: number;
    deadLetterCount: number;
    failureRate: number;
    stateCounts: {
      queued: number;
      cued: number;
      cueOnly: number;
      scheduled: number;
      projecting: number;
      projected: number;
      merged: number;
      failed: number;
      deadLetter: number;
    };
  };
  recall: {
    status: LifecycleStageStatus;
    activeEntityCount: number;
    topScore: number;
    triggerCount: number;
    intentions: {
      activeCount: number;
      refreshContextCount: number;
      afterConsolidationCount: number;
      pinnedResultCount: number;
      needsRefreshCount: number;
      latestRefreshedAt: string | null;
    };
    topActivated: Array<{
      id: string;
      name: string;
      entityType: string;
      summary: string | null;
      activation: number;
      accessCount: number;
    }>;
  };
  consolidate: {
    status: LifecycleStageStatus;
    isRunning: boolean;
    schedulerActive: boolean;
    cycleCount: number;
    pressure: {
      value: number;
      threshold: number;
      episodesSinceLast: number;
      entitiesCreated: number;
      lastCycleTime: number | null;
    } | null;
    latestCycle: ConsolidationCycleSummary | null;
  };
  recentEpisodes: Episode[];
}

export type WsReadyState = "connecting" | "connected" | "reconnecting" | "disconnected";

export interface TimeRange {
  start: string;
  end: string;
}

export type GraphRenderMode = "3d" | "2d";
export type DashboardMode = "observatory" | "nerve";

export type BrainLoopEvaluationSignalKey =
  | "cueUsefulness"
  | "projectionYield"
  | "recallQuality"
  | "falseRecall"
  | "triageCalibration"
  | "consolidationEffect";

export interface BrainLoopEvaluationSignal {
  status: string;
  evidenceCount: number;
  metric: number | null;
  gap: string | null;
}

export interface HumanLabelEvidence {
  status: string;
  artifactPath: string | null;
  artifactSha256: string | null;
  kind: string | null;
  source: string | null;
  client: string | null;
  capturedAt: string | null;
  sessionId: string | null;
  labeler: string | null;
  humanLabeled: boolean;
  recallSampleCount: number;
  sessionSampleCount: number;
  minRecallSamples: number;
  minSessionSamples: number;
  sampleSources: string[];
  failures: string[];
}

export interface AdoptionEvidence {
  status: string;
  artifactPath: string | null;
  artifactSha256: string | null;
  adoptionStatus: string | null;
  authorityPath: string | null;
  callsPath: string | null;
  callCount: number;
  client: string | null;
  requiredClient: string | null;
  gateRequiredClient: string | null;
  capturedAt: string | null;
  sessionId: string | null;
  sessionFilter: string | null;
  source: string | null;
  requiredLiveEvidence: boolean;
  blockers?: string[];
  blockerDetails?: string[];
  mcpServerFailures?: string[];
  requiredTools: {
    expected: string[];
    observed: string[];
    missing: string[];
    inOrder: boolean;
  };
  capture: {
    destination: string | null;
    expectedTool: string | null;
    observedTools: string[];
    missing: boolean;
  };
  fileMemory: {
    present: boolean | null;
    substitutedForEngram: boolean;
  };
  failures: string[];
}

export interface AdoptionClientEvidenceReport {
  client: string | null;
  requiredClient: string | null;
  status: string;
  artifactPath: string | null;
  artifactSha256: string | null;
  capturedAt: string | null;
  sessionId: string | null;
  blockers?: string[];
  blockerDetails?: string[];
  mcpServerFailures?: string[];
  failures: string[];
}

export interface AdoptionClientEvidence {
  status: string;
  requiredClients: string[];
  observedClients: string[];
  reportCount: number;
  reports: AdoptionClientEvidenceReport[];
  blockers?: string[];
  mcpServerFailures?: string[];
  failures: string[];
}

export interface ReleaseEvidenceComponent {
  status: string;
  missing: string[];
  failures: string[];
  requiredClients?: string[];
  observedClients?: string[];
  blockers?: string[];
  blockerDetails?: string[];
  mcpServerFailures?: string[];
}

export interface ReleaseEvidenceSummary {
  status: string;
  components: {
    evaluationSignals: ReleaseEvidenceComponent;
    humanLabels: ReleaseEvidenceComponent;
    adoption: ReleaseEvidenceComponent;
    adoptionClients: ReleaseEvidenceComponent;
  };
  missing: string[];
  failures: string[];
}

export interface BrainLoopEvaluationReport {
  groupId: string;
  generatedAt: string;
  loop: LifecycleStageKey[];
  totals: {
    episodes: number;
    entities: number;
    relationships: number;
    activeEntities: number;
  };
  capture: {
    status: string;
    episodeCount: number;
    activeCount: number;
  };
  cue: {
    status: string;
    cueCount: number;
    episodesWithoutCues: number;
    coverage: number;
    hitCount: number;
    hitEpisodeCount: number;
    hitEpisodeRate: number;
    surfacedCount: number;
    selectedCount: number;
    usedCount: number;
    nearMissCount: number;
    selectedRate: number;
    usedRate: number;
    nearMissRate: number;
    avgPolicyScore: number;
    projectionConversionRate: number;
  };
  project: {
    status: string;
    stateCounts: {
      queued: number;
      cued: number;
      cueOnly: number;
      scheduled: number;
      projecting: number;
      projected: number;
      merged: number;
      failed: number;
      deadLetter: number;
    };
    trackedCount: number;
    projectedCount: number;
    activeCount: number;
    projectedRate: number;
    backlogRate: number;
    failedCount: number;
    deadLetterCount: number;
    attemptedEpisodeCount: number;
    totalAttempts: number;
    failureRate: number;
    avgProcessingDurationMs: number;
    avgTimeToProjectionMs: number;
    yield: {
      linkedEntityCount: number;
      relationshipCount: number;
      avgLinkedEntitiesPerProjectedEpisode: number;
      avgRelationshipsPerProjectedEpisode: number;
    };
  };
  recall: {
    status: string;
    totalAnalyses: number;
    triggerCount: number;
    runtimeFalseRecallRate: number;
    runtimeSurfacedToUsedRatio: number | null;
    graphLiftRate: number;
    probeTriggerRate: number;
    latency: {
      analyzerMs: {
        avgMs: number;
        p95Ms: number;
      };
      probeMs: {
        avgMs: number;
        p95Ms: number;
      };
    };
    control: {
      usedCount: number;
      dismissedCount: number;
      surfacedCount: number;
      selectedCount: number;
      confirmedCount: number;
      correctedCount: number;
      graphOverrideCount: number;
      adaptiveThresholdsEnabled: boolean;
      thresholds: {
        linguistic: number;
        borderline: number;
        resonance: number;
      };
    };
    familyContributions: Record<string, number>;
    evaluation: {
      status: string;
      sampleCount: number;
      needStatus: string;
      needLabeledCount: number;
      neededCount: number;
      missedCount: number;
      memoryNeedPrecision: number | null;
      memoryNeedRecall: number | null;
      missedRecallRate: number | null;
      usefulPacketRate: number | null;
      falseRecallRate: number | null;
      surfacedCount: number;
      usedCount: number;
      surfacedToUsedRatio: number | null;
    };
    continuity: {
      status: string;
      sampleCount: number;
      sessionContinuityLift: number | null;
      openLoopRecoveryRate: number | null;
      temporalCorrectness: number | null;
    };
  };
  consolidate: {
    status: string;
    cycleCount: number;
    latestStatus: string | null;
    latestCycle: Record<string, unknown> | null;
    phaseStatusCounts: Record<string, number>;
    phaseTotals: Record<
      string,
      { runs: number; itemsProcessed: number; itemsAffected: number; effectRate: number }
    >;
    adjudication: {
      status: string;
      phaseCount: number;
      runs: number;
      itemsProcessed: number;
      itemsAffected: number;
      itemsUnaffected: number;
      effectRate: number;
      errorCount: number;
      openEvidenceCount?: number;
      openRequestCount?: number;
      openWorkCount?: number;
      pendingEvidenceCount?: number;
      deferredEvidenceCount?: number;
      approvedEvidenceCount?: number;
      pendingRequestCount?: number;
      deferredRequestCount?: number;
      errorRequestCount?: number;
      evidenceStatusCounts?: Record<string, number>;
      requestStatusCounts?: Record<string, number>;
      phaseTotals: Record<
        string,
        { runs: number; itemsProcessed: number; itemsAffected: number; effectRate: number }
      >;
    };
    calibration: {
      status: string;
      snapshotCount: number;
      phaseTotals: Record<
        string,
        {
          snapshots: number;
          totalTraces: number;
          labeledExamples: number;
          oracleExamples: number;
          abstainCount: number;
          accuracy: number | null;
          meanConfidence: number | null;
          expectedCalibrationError: number | null;
        }
      >;
    };
    itemsProcessed: number;
    itemsAffected: number;
    effectRate: number;
    errorCount: number;
  };
  evaluationSignals: Record<BrainLoopEvaluationSignalKey, BrainLoopEvaluationSignal>;
  coverageGaps: string[];
  releaseEvidence?: ReleaseEvidenceSummary | null;
  humanLabelEvidence?: HumanLabelEvidence | null;
  adoptionEvidence?: AdoptionEvidence | null;
  additionalAdoptionEvidence?: AdoptionEvidence[];
  adoptionClientEvidence?: AdoptionClientEvidence | null;
}

export interface RecallEvaluationInput {
  recallTriggered: boolean;
  recallHelped: boolean;
  recallNeeded?: boolean | null;
  packetsSurfaced: number;
  packetsUsed: number;
  falseRecalls: number;
  source?: string;
  query?: string | null;
  notes?: string | null;
}

export interface RecallEvaluationWriteResponse {
  status: string;
  groupId: string;
  sample: {
    id: string;
    recallTriggered: boolean;
    recallHelped: boolean;
    recallNeeded: boolean | null;
    packetsSurfaced: number;
    packetsUsed: number;
    falseRecalls: number;
    source: string;
    query: string | null;
    notes: string | null;
    timestamp: number;
  };
}

export interface SessionContinuityEvaluationInput {
  baselineScore: number;
  memoryScore: number;
  openLoopExpected: boolean;
  openLoopRecovered: boolean;
  temporalExpected: boolean;
  temporalCorrect: boolean;
  source?: string;
  scenario?: string | null;
  notes?: string | null;
}

export interface SessionContinuityEvaluationWriteResponse {
  status: string;
  groupId: string;
  sample: {
    id: string;
    baselineScore: number;
    memoryScore: number;
    openLoopExpected: boolean;
    openLoopRecovered: boolean;
    temporalExpected: boolean;
    temporalCorrect: boolean;
    source: string;
    scenario: string | null;
    notes: string | null;
    timestamp: number;
  };
}

export type DashboardView =
  | "lifecycle"
  | "knowledge"
  | "graph"
  | "timeline"
  | "feed"
  | "activation"
  | "stats"
  | "evaluation"
  | "consolidation"
  | "profile"
  | "synaptic_log"
  | "neural_field"
  | "ingestion"
  | "nerve_center"
  | "adjudicate"
  | "immunity";

export interface RecallResultEntity {
  resultType: "entity";
  entity: { id: string; name: string; entityType: string; summary: string };
  score: number;
  scoreBreakdown: { semantic: number; activation: number; edgeProximity: number; explorationBonus: number };
  relationships: Array<{ predicate: string; source_id: string; target_id: string; weight: number }>;
}

export interface RecallResultEpisode {
  resultType: "episode";
  episode: { id: string; content: string; source: string; createdAt: string };
  score: number;
  scoreBreakdown: { semantic: number; activation: number; edgeProximity: number; explorationBonus: number };
}

export interface RecallResultCueEpisode {
  resultType: "cue_episode";
  cue: {
    episodeId: string;
    cueText: string | null;
    supportingSpans: string[];
    projectionState: EpisodeProjectionState | null;
    routeReason: string | null;
    hitCount: number;
    surfacedCount: number;
    selectedCount: number;
    usedCount: number;
    nearMissCount: number;
    policyScore: number | null;
    lastFeedbackAt: string | null;
    lastProjectedAt: string | null;
  };
  episode: { id: string | null; source: string | null; createdAt: string | null };
  score: number;
  scoreBreakdown: { semantic: number; activation: number; edgeProximity: number; explorationBonus: number };
}

export type RecallResult = RecallResultEntity | RecallResultEpisode | RecallResultCueEpisode;

export interface RecallResponseLifecycle {
  stage: "recall";
  recallMode: "explicit";
  resultCount: number;
  packetCount: number;
}

export interface RecallResponse {
  operation: "recall";
  lifecycle: RecallResponseLifecycle;
  items: RecallResult[];
  packets?: Array<Record<string, unknown>>;
  query: string;
}

export interface FactResult {
  subject: string;
  predicate: string;
  object: string;
  validFrom: string | null;
  validTo: string | null;
  confidence: number | null;
}

export type IntentMode = "asking" | "remembering" | "observing" | "forgetting" | null;

export interface PulseEntity {
  entityId: string;
  name: string;
  entityType: string;
  currentActivation: number;
}

export interface ConfirmDialogState {
  type: "delete" | "forget";
  entityId?: string;
  entityName: string;
  title: string;
  message: string;
}

export interface GraphDelta {
  nodesAdded?: GraphNode[];
  nodesUpdated?: GraphNode[];
  nodesRemoved?: string[];
  edgesAdded?: GraphEdge[];
  edgesRemoved?: string[];
}

export interface GraphSlice {
  nodes: Record<string, GraphNode>;
  edges: Record<string, GraphEdge>;
  centerNodeId: string | null;
  brainMapScope: BrainMapScope;
  representation: GraphRepresentationMeta | null;
  atlas: AtlasData | null;
  activeRegionId: string | null;
  regionData: RegionData | null;
  isLoading: boolean;
  error: string | null;
  loadAtlas: (
    options?: { refresh?: boolean; snapshotId?: string | null },
  ) => Promise<void>;
  loadRegion: (
    regionId: string,
    options?: { refresh?: boolean; snapshotId?: string | null },
  ) => Promise<void>;
  loadNeighborhood: (
    centerId?: string,
    depth?: number,
    options?: { regionId?: string | null },
  ) => Promise<void>;
  loadInitialGraph: () => Promise<void>;
  expandNode: (nodeId: string) => Promise<void>;
  mergeGraphDelta: (delta: GraphDelta) => void;
  updateNodeActivations: (updates: Array<{ entityId: string; activation: number }>) => void;
  loadGraphAt: (timestamp: string, centerId?: string) => Promise<void>;
  clear: () => void;
}

export interface SelectionSlice {
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  selectedEdgeId: string | null;
  searchQuery: string;
  searchResults: SearchResult[];
  isSearching: boolean;
  searchError: string | null;
  entityHistory: string[];
  entityHistoryIndex: number;
  isNavigatingHistory: boolean;
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  setSearchQuery: (query: string) => void;
  executeSearch: (query: string) => Promise<void>;
  clearSearch: () => void;
  pushEntityHistory: (nodeId: string) => void;
  entityGoBack: () => string | null;
  entityGoForward: () => string | null;
  canEntityGoBack: () => boolean;
  canEntityGoForward: () => boolean;
}

export interface PreferencesSlice {
  currentView: DashboardView;
  lifecycleDrilldownStage: LifecycleStageKey | null;
  renderMode: GraphRenderMode;
  showActivationHeatmap: boolean;
  showEdgeLabels: boolean;
  showFpsOverlay: boolean;
  darkMode: boolean;
  graphMaxNodes: number;
  lastAtlasVisitAt: string | null;
  lastAtlasSnapshotId: string | null;
  dashboardMode: DashboardMode;
  setCurrentView: (view: DashboardView) => void;
  setLifecycleDrilldownStage: (stage: LifecycleStageKey | null) => void;
  setRenderMode: (mode: GraphRenderMode) => void;
  toggleActivationHeatmap: () => void;
  toggleEdgeLabels: () => void;
  toggleFpsOverlay: () => void;
  toggleDarkMode: () => void;
  setGraphMaxNodes: (n: number) => void;
  recordAtlasVisit: (
    visit: { generatedAt: string | null; snapshotId?: string | null },
  ) => void;
  setDashboardMode: (mode: DashboardMode) => void;
}

export interface TimeSlice {
  timePosition: string | null;
  timeRange: TimeRange | null;
  isTimeScrubbing: boolean;
  atlasSnapshotId: string | null;
  atlasHistory: AtlasHistoryEntry[];
  setTimePosition: (ts: string | null) => void;
  setTimeRange: (range: TimeRange | null) => void;
  setIsTimeScrubbing: (v: boolean) => void;
  setAtlasSnapshotId: (snapshotId: string | null) => void;
  setAtlasHistory: (history: AtlasHistoryEntry[]) => void;
}

export interface EpisodeSlice {
  episodes: Episode[];
  episodeCursor: string | null;
  hasMoreEpisodes: boolean;
  isLoadingEpisodes: boolean;
  isIngesting: boolean;
  loadEpisodes: (cursor?: string) => Promise<void>;
  prependEpisode: (episode: Episode) => void;
  setIngesting: (isIngesting: boolean) => void;
  updateEpisodeStatus: (episodeId: string, status: EpisodeStatus, error?: string | null) => void;
}

export interface StatsSlice {
  stats: GraphStats | null;
  isLoadingStats: boolean;
  loadStats: () => Promise<void>;
}

export interface LifecycleSlice {
  lifecycleSummary: LifecycleSummary | null;
  isLoadingLifecycleSummary: boolean;
  loadLifecycleSummary: () => Promise<void>;
}

export interface EvaluationSlice {
  evaluationReport: BrainLoopEvaluationReport | null;
  isLoadingEvaluationReport: boolean;
  isSavingRecallEvaluation: boolean;
  isSavingSessionEvaluation: boolean;
  loadEvaluationReport: () => Promise<void>;
  recordRecallEvaluation: (input: RecallEvaluationInput) => Promise<void>;
  recordSessionContinuityEvaluation: (input: SessionContinuityEvaluationInput) => Promise<void>;
}

export interface WebSocketSlice {
  readyState: WsReadyState;
  lastSeq: number;
  reconnectAttempt: number;
  setReadyState: (state: WsReadyState) => void;
  setLastSeq: (seq: number) => void;
  setReconnectAttempt: (n: number) => void;
}

export interface ActivationItem {
  entityId: string;
  name: string;
  entityType: string;
  currentActivation: number;
  accessCount: number;
  lastAccessedAt: string | null;
  decayRate: number;
}

export interface CurvePoint {
  timestamp: string;
  activation: number;
}

export interface ActivationPulse {
  entityId: string;
  name: string;
  entityType: string;
  activation: number;
  accessedVia: string;
  timestamp: number;
  cascadeIntensity?: number;
}

export interface ActivationSlice {
  activationLeaderboard: ActivationItem[];
  selectedActivationEntity: string | null;
  decayCurve: CurvePoint[];
  decayFormula: string;
  accessEvents: string[];
  isActivationSubscribed: boolean;
  isLoadingCurve: boolean;
  activationPulses: ActivationPulse[];
  setActivationLeaderboard: (items: ActivationItem[]) => void;
  selectActivationEntity: (id: string | null) => void;
  loadDecayCurve: (entityId: string) => Promise<void>;
  setIsActivationSubscribed: (v: boolean) => void;
  addActivationPulse: (pulse: Omit<ActivationPulse, "timestamp">) => void;
  clearExpiredPulses: (maxAgeMs?: number) => void;
}

export interface ConsolidationPhaseResult {
  phase: string;
  status: "success" | "skipped" | "error";
  items_processed: number;
  items_affected: number;
  duration_ms: number;
  error: string | null;
}

export interface ConsolidationCycleSummary {
  id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  error?: string | null;
  phase_issue?: string | null;
  dry_run: boolean;
  trigger: string;
  started_at: number;
  completed_at: number | null;
  total_duration_ms: number;
  phases: ConsolidationPhaseResult[];
  summary?: {
    total_processed: number;
    total_affected: number;
    description: string;
  };
}

export interface ConsolidationMerge {
  id: string;
  keep_name: string;
  remove_name: string;
  similarity: number;
  decision_confidence?: number | null;
  decision_source?: string | null;
  decision_reason?: string | null;
  relationships_transferred: number;
}

export interface ConsolidationIdentifierReview {
  id: string;
  entity_a_name: string;
  entity_b_name: string;
  entity_a_type: string;
  entity_b_type: string;
  raw_similarity: number;
  adjusted_similarity?: number | null;
  decision_source?: string | null;
  decision_reason?: string | null;
  entity_a_regime?: string | null;
  entity_b_regime?: string | null;
  canonical_identifier_a?: string | null;
  canonical_identifier_b?: string | null;
  review_status: string;
  metadata?: Record<string, unknown>;
}

export interface ConsolidationInferredEdge {
  id: string;
  source_name: string;
  target_name: string;
  co_occurrence_count: number;
  confidence: number;
}

export interface ConsolidationPrune {
  id: string;
  entity_name: string;
  entity_type: string;
  reason: string;
}

export interface ConsolidationDream {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  weight_delta: number;
}

export interface ConsolidationReplay {
  id: string;
  episode_id: string;
  skipped_reason: string | null;
  new_entities_found: number;
  new_relationships_found: number;
  entities_updated: number;
}

export interface ConsolidationReindex {
  id: string;
  entity_name: string;
  source_phase: string;
}

export interface ConsolidationCycleDetail extends ConsolidationCycleSummary {
  error: string | null;
  merges: ConsolidationMerge[];
  identifier_reviews: ConsolidationIdentifierReview[];
  inferred_edges: ConsolidationInferredEdge[];
  prunes: ConsolidationPrune[];
  dreams: ConsolidationDream[];
  replays: ConsolidationReplay[];
  reindexes: ConsolidationReindex[];
}

export interface ConsolidationPressure {
  value: number;
  threshold: number;
  episodes_since_last: number;
  entities_created: number;
}

export interface KnowledgeSlice {
  // Existing state
  knowledgeQuery: string;
  knowledgeResults: RecallResult[];
  isRecalling: boolean;
  activeTypeFilter: string | null;
  entityGroups: Record<string, SearchResult[]>;
  isLoadingEntities: boolean;
  expandedEntityId: string | null;
  entityDetail: EntityDetail | null;
  inputText: string;
  isSending: boolean;

  // New state
  pulseEntities: PulseEntity[];
  isPulseLoading: boolean;
  drawerEntityId: string | null;
  drawerEntity: EntityDetail | null;
  isDrawerLoading: boolean;
  searchOverlayOpen: boolean;
  browseOverlayOpen: boolean;
  intentMode: IntentMode;
  confirmDialog: ConfirmDialogState | null;

  // Existing actions
  setKnowledgeQuery: (q: string) => void;
  executeRecall: (query: string) => Promise<void>;
  loadEntityGroups: () => Promise<void>;
  setActiveTypeFilter: (type: string | null) => void;
  expandEntity: (id: string | null) => Promise<void>;
  setInputText: (t: string) => void;
  submitInput: (text: string, appendMessages?: (userText: string, assistantText: string) => void) => Promise<void>;

  // New actions
  loadPulseEntities: () => Promise<void>;
  setPulseEntities: (entities: PulseEntity[]) => void;
  openDrawer: (id: string) => Promise<void>;
  closeDrawer: () => void;
  setSearchOverlayOpen: (open: boolean) => void;
  setBrowseOverlayOpen: (open: boolean) => void;
  updateEntity: (id: string, patch: { name?: string; summary?: string }) => Promise<void>;
  deleteEntity: (id: string) => Promise<void>;
  setConfirmDialog: (dialog: ConfirmDialogState | null) => void;
  confirmAction: () => Promise<void>;

  // Intention state (Wave 4 v2)
  intentions: IntentionItem[];
  handleIntentionEvent: (data: Record<string, unknown>) => void;
  loadIntentions: () => Promise<void>;
}

export interface IntentionItem {
  id: string;
  triggerText: string;
  actionText: string;
  triggerType: string;
  threshold: number;
  fireCount: number;
  maxFires: number;
  enabled: boolean;
  priority: string;
  warmthRatio: number;
  linkedEntityIds: string[];
}

export type NeuralSpecialization = "Architect" | "Synthesizer" | "Narrator" | "Integrator" | "Biochemist" | "Topologist" | "Polymath";

export interface CerebralStats {
  level: number;
  plasticity: number;
  plasticityToNext: number;
  homeostasis: number;
  morale: number;
  synapticCredits: number;
  specialization: NeuralSpecialization;
  domainScores: Record<string, number>;
}

export interface SynapticEvent {
  id: string;
  text: string;
  plasticity: number;
  timestamp: number;
}

export type NeuralFieldLayer = "activity" | "clusters" | "heatmap" | "entropy";

export interface NerveCenterSlice {
  dashboardMode: DashboardMode;
  cerebralStats: CerebralStats;
  synapticEvents: SynapticEvent[];
  feedbackPositive: number;
  feedbackNegative: number;
  notifications: MemoryNotification[];
  adjudicationRequests: AdjudicationRequestData[];
  isAdjudicating: boolean;
  selectedNeuralLayer: NeuralFieldLayer;
  setDashboardMode: (mode: DashboardMode) => void;
  computeCerebralStats: (stats: GraphStats) => void;
  addSynapticEvent: (text: string, plasticity: number) => void;
  recordFeedback: (positive: boolean) => void;
  loadNotifications: () => Promise<void>;
  dismissNotifications: (ids: string[]) => Promise<void>;
  loadAdjudications: () => Promise<void>;
  resolveAdjudication: (body: AdjudicateBody) => Promise<void>;
  setSelectedNeuralLayer: (layer: NeuralFieldLayer) => void;
}

export interface ConsolidationSlice {
  cycles: ConsolidationCycleSummary[];
  isLoadingCycles: boolean;
  selectedCycleId: string | null;
  cycleDetail: ConsolidationCycleDetail | null;
  isLoadingDetail: boolean;
  isRunning: boolean;
  schedulerActive: boolean;
  pressure: ConsolidationPressure | null;
  triggerDryRun: boolean;
  setTriggerDryRun: (value: boolean) => void;
  loadStatus: () => Promise<void>;
  loadCycles: () => Promise<void>;
  selectCycle: (id: string | null) => void;
  loadCycleDetail: (id: string) => Promise<void>;
  triggerCycle: (dryRun: boolean) => Promise<void>;
}

export interface ConversationSlice {
  conversations: Array<{
    id: string;
    title: string | null;
    sessionDate: string;
    createdAt: string;
    updatedAt: string;
    entityIds: string[];
  }>;
  isLoadingConversations: boolean;
  activeConversationId: string | null;
  conversationSidebarOpen: boolean;
  loadConversations: () => Promise<void>;
  setActiveConversation: (id: string | null) => void;
  setConversationId: (id: string | null) => void;
  toggleConversationSidebar: () => void;
  startNewConversation: () => void;
}

export type EngramStore =
  GraphSlice &
  SelectionSlice &
  PreferencesSlice &
  TimeSlice &
  EpisodeSlice &
  StatsSlice &
  LifecycleSlice &
  EvaluationSlice &
  WebSocketSlice &
  ActivationSlice &
  ConsolidationSlice &
  KnowledgeSlice &
  ConversationSlice &
  NerveCenterSlice;

export interface MemoryNotification {
  id: string;
  group_id: string;
  notification_type:
    | "temporal_intention"
    | "dream_association"
    | "schema_discovery"
    | "entity_maturation"
    | "entity_merge"
    | "activation_anomaly"
    | "immunity_sweep";
  priority: "low" | "normal" | "high";
  title: string;
  body: string;
  entity_ids: string[];
  metadata: Record<string, any>;
  source_cycle_id: string | null;
  created_at: number;
  dismissed_at: number | null;
  surfaced_count: number;
}

export interface AdjudicateBody {
  request_id: string;
  entities?: any[];
  relationships?: any[];
  reject_evidence_ids?: string[];
  model_tier?: string;
  rationale?: string;
}

export interface AdjudicationResolutionOutcome {
  status: string;
  requestId: string;
  committedIds: Record<string, string>;
  supersededEvidenceIds: string[];
  replacementEvidenceIds: string[];
}

export interface AdjudicationRequestData {
  request_id: string;
  episode_id: string;
  ambiguity_tags: string[];
  selected_text: string;
  candidate_evidence: Array<{
    evidence_id: string;
    fact_class: string;
    payload: any;
  }>;
  instructions: string;
}
