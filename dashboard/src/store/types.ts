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
  | "graph"
  | "timeline"
  | "feed"
  | "activation"
  | "stats";

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
  loadGraphAt: (timestamp: string) => Promise<void>;
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

export interface ActivationSlice {
  activationLeaderboard: ActivationItem[];
  selectedActivationEntity: string | null;
  decayCurve: CurvePoint[];
  decayFormula: string;
  accessEvents: string[];
  isActivationSubscribed: boolean;
  isLoadingCurve: boolean;
  setActivationLeaderboard: (items: ActivationItem[]) => void;
  selectActivationEntity: (id: string | null) => void;
  loadDecayCurve: (entityId: string) => Promise<void>;
  setIsActivationSubscribed: (v: boolean) => void;
}

export type EngramStore =
  GraphSlice &
  SelectionSlice &
  PreferencesSlice &
  TimeSlice &
  EpisodeSlice &
  StatsSlice &
  WebSocketSlice &
  ActivationSlice;
