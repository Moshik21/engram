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
  GraphRepresentationMeta,
  ConsolidationCycleSummary,
  ConsolidationCycleDetail,
  ConsolidationPressure,
  RecallResult,
  FactResult,
  IntentionItem,
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

export const api = {
  getHealth: () => fetchJSON<HealthResponse>("/health"),

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
    fetchJSON<{ status: string; cycle_id: string }>("/api/consolidation/trigger?dry_run=" + dryRun, { method: "POST" }),

  // Knowledge API
  recall: (params: { q: string; limit?: number }) => {
    const sp = new URLSearchParams();
    sp.set("q", params.q);
    if (params.limit) sp.set("limit", String(params.limit));
    return fetchJSON<{ items: RecallResult[]; query: string }>(`/api/knowledge/recall?${sp}`);
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
};
