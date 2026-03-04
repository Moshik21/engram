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

export interface SearchResult {
  id: string;
  name: string;
  entityType: string;
  summary: string | null;
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
  activationCurrent: number;
  accessCount: number;
  lastAccessed: string | null;
  createdAt: string;
  updatedAt: string;
  facts: EntityFact[];
}

export type EpisodeStatus = "queued" | "processing" | "completed" | "failed";

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

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ name: string; entityType?: string; score?: number }>;
  timestamp: number;
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
  isLoading: boolean;
  error: string | null;
  loadNeighborhood: (centerId?: string, depth?: number) => Promise<void>;
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
  darkMode: boolean;
  graphMaxNodes: number;
  setCurrentView: (view: DashboardView) => void;
  setRenderMode: (mode: GraphRenderMode) => void;
  toggleActivationHeatmap: () => void;
  toggleEdgeLabels: () => void;
  toggleDarkMode: () => void;
  setGraphMaxNodes: (n: number) => void;
}

export interface TimeSlice {
  timePosition: string | null;
  timeRange: TimeRange | null;
  isTimeScrubbing: boolean;
  setTimePosition: (ts: string | null) => void;
  setTimeRange: (range: TimeRange | null) => void;
  setIsTimeScrubbing: (v: boolean) => void;
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
  relationships_transferred: number;
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
  chatMessages: ChatMessage[];
  isChatStreaming: boolean;
  chatOpen: boolean;
  setKnowledgeQuery: (q: string) => void;
  executeRecall: (query: string) => Promise<void>;
  loadEntityGroups: () => Promise<void>;
  setActiveTypeFilter: (type: string | null) => void;
  expandEntity: (id: string | null) => Promise<void>;
  setInputText: (t: string) => void;
  submitInput: (text: string) => Promise<void>;
  sendChatMessage: (message: string) => Promise<void>;
  toggleChat: () => void;
  clearChat: () => void;
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
  loadStatus: () => Promise<void>;
  loadCycles: () => Promise<void>;
  selectCycle: (id: string | null) => void;
  loadCycleDetail: (id: string) => Promise<void>;
  triggerCycle: (dryRun: boolean) => Promise<void>;
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
  KnowledgeSlice;
