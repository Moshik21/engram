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

export type EpisodeStatus = "queued" | "processing" | "extracting" | "completed" | "failed";

export interface Episode {
  episodeId: string;
  content: string;
  source: string;
  status: EpisodeStatus;
  createdAt: string;
  updatedAt: string;
  entities: Array<{ id: string; name: string; entityType: string }>;
  factsCount: number;
  processingDurationMs: number | null;
  error: string | null;
  retryCount: number;
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
  topActivated: Array<{ id: string; name: string; entityType: string; activation: number }>;
  topConnected: Array<{ id: string; name: string; entityType: string; connectionCount: number }>;
  growthTimeline: Array<{ date: string; entities: number; episodes: number }>;
}

export type WsReadyState = "connecting" | "connected" | "reconnecting" | "disconnected";

export interface TimeRange {
  start: string;
  end: string;
}

export type GraphRenderMode = "3d" | "2d";
export type DashboardView =
  | "knowledge"
  | "graph"
  | "timeline"
  | "feed"
  | "activation"
  | "stats"
  | "consolidation";

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

export type RecallResult = RecallResultEntity | RecallResultEpisode;

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
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  setSearchQuery: (query: string) => void;
  executeSearch: (query: string) => Promise<void>;
  clearSearch: () => void;
}

export interface PreferencesSlice {
  currentView: DashboardView;
  renderMode: GraphRenderMode;
  showActivationHeatmap: boolean;
  showEdgeLabels: boolean;
  showFpsOverlay: boolean;
  darkMode: boolean;
  graphMaxNodes: number;
  lastAtlasVisitAt: string | null;
  lastAtlasSnapshotId: string | null;
  setCurrentView: (view: DashboardView) => void;
  setRenderMode: (mode: GraphRenderMode) => void;
  toggleActivationHeatmap: () => void;
  toggleEdgeLabels: () => void;
  toggleFpsOverlay: () => void;
  toggleDarkMode: () => void;
  setGraphMaxNodes: (n: number) => void;
  recordAtlasVisit: (
    visit: { generatedAt: string | null; snapshotId?: string | null },
  ) => void;
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
  loadEpisodes: (cursor?: string) => Promise<void>;
  prependEpisode: (episode: Episode) => void;
  updateEpisodeStatus: (episodeId: string, status: EpisodeStatus, error?: string | null) => void;
}

export interface StatsSlice {
  stats: GraphStats | null;
  isLoadingStats: boolean;
  loadStats: () => Promise<void>;
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
  dry_run: boolean;
  trigger: string;
  started_at: number;
  completed_at: number | null;
  total_duration_ms: number;
  phases: ConsolidationPhaseResult[];
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
  WebSocketSlice &
  ActivationSlice &
  ConsolidationSlice &
  KnowledgeSlice &
  ConversationSlice;
