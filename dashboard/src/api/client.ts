import type {
  GraphNode,
  GraphEdge,
  SearchResult,
  EntityDetail,
  Episode,
  GraphStats,
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
  truncated: boolean;
  totalInNeighborhood: number;
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

  searchEntities: async (params: { q?: string; type?: string; limit?: number }): Promise<SearchResult[]> => {
    const sp = new URLSearchParams();
    if (params.q) sp.set("q", params.q);
    if (params.type) sp.set("type", params.type);
    if (params.limit) sp.set("limit", String(params.limit));
    const data = await fetchJSON<{ items: Array<{ id: string; name: string; entityType: string; summary: string | null; activationCurrent: number }>; total: number }>(`/api/entities/search?${sp}`);
    return data.items.map((item) => ({
      id: item.id,
      name: item.name,
      entityType: item.entityType,
      summary: item.summary,
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const raw: any = await fetchJSON("/api/stats");
    const s = raw.stats ?? {};
    return {
      totalEntities: s.entities ?? 0,
      totalRelationships: s.relationships ?? 0,
      totalEpisodes: s.episodes ?? 0,
      entityTypeCounts: s.entity_type_distribution ?? {},
      topActivated: (raw.topActivated ?? []).map((a: any) => ({
        id: a.id,
        name: a.name,
        entityType: a.entityType,
        activation: a.activationCurrent ?? a.activation ?? 0,
      })),
      topConnected: (raw.topConnected ?? []).map((c: any) => ({
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
    return fetchJSON<NeighborhoodResponse>(`/api/graph/neighborhood?${sp}`);
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

  getConversationMessages: (convId: string) =>
    fetchJSON<{ messages: ConversationMessage[] }>(`/api/conversations/${convId}/messages`),

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
