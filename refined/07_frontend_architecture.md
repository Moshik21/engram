# 07 -- Frontend Architecture & API Contract

> The dashboard is the product's face. This document defines the state management,
> API surface, WebSocket protocol, and component architecture that make it work.

---

## 1. Problems Identified in Original Spec

The Tech_Spec.md dashboard section (Section 5) has four gaps that would block implementation:

1. **No state management.** The spec lists React hooks (`useGraph`, `useWebSocket`, `useActivation`) but never defines what state they hold, how it's shared between components, or how concurrent WebSocket updates merge with REST-fetched data. Without a store, every component re-fetches independently.

2. **Full-graph endpoint won't scale.** `GET /graph` returns the entire graph. A personal memory graph hits 10K+ nodes within months. react-force-graph-3d handles thousands of nodes in theory, but downloading and parsing a 2MB JSON payload on every page load -- and on every WebSocket `graph.updated` event -- creates unacceptable latency and jank.

3. **Time-scrubber query undefined.** The timeline view mentions "scrub through time to see how the graph evolved" but the spec provides no endpoint, no query parameters, and no definition of what "point-in-time state" means for a bi-temporal graph. The frontend has nothing to call.

4. **WebSocket contract missing.** The spec says "WebSocket for live updates" without defining event types, payload schemas, ordering guarantees, or reconnection behavior. The ingestion pipeline doc (`03_async_ingestion.md`) defines episode-lifecycle events but the dashboard needs graph-mutation events, activation-change events, and a framing protocol.

---

## 2. State Management: Zustand Store

### 2.1 Why Zustand

- **Small bundle.** ~1KB gzipped. No providers, no context wrappers.
- **TypeScript-first.** Full inference on slices and selectors.
- **Subscriptions are surgical.** Components subscribe to specific slices, not the whole tree. A node tooltip re-renders on `selectedNode` change; the graph canvas re-renders on `nodes`/`edges` change. No wasted renders.
- **Middleware.** `devtools` for debugging, `immer` for immutable updates on nested graph data, `persist` for saving user preferences to localStorage.
- **Compatible with Vercel AI SDK.** Zustand stores can be updated from `useChat` / streaming callbacks without lifecycle conflicts.

### 2.2 Store Shape (TypeScript)

```typescript
// dashboard/src/store/types.ts

// ─── Graph Data ─────────────────────────────────────────────
export interface GraphNode {
  id: string;
  name: string;
  entityType: EntityType;
  summary: string;
  activationBase: number;    // 0.0 - 1.0
  activationCurrent: number; // 0.0 - 1.0, computed at query time
  accessCount: number;
  lastAccessed: string;      // ISO 8601
  createdAt: string;
  updatedAt: string;
  // Layout hints (set by force simulation, persisted across re-renders)
  x?: number;
  y?: number;
  z?: number;
}

export type EntityType =
  | "Person"
  | "Organization"
  | "Project"
  | "Concept"
  | "Preference"
  | "Location"
  | "Event"
  | "Tool";

export interface GraphEdge {
  id: string;
  source: string;          // entity ID
  target: string;          // entity ID
  predicate: string;       // "works_at", "interested_in", etc.
  weight: number;          // 0.0 - 1.0
  validFrom: string;       // ISO 8601
  validTo: string | null;  // null = still valid
  createdAt: string;
}

// ─── Episode / Feed ─────────────────────────────────────────
export interface Episode {
  episodeId: string;
  content: string;
  source: string;
  status: EpisodeStatus;
  createdAt: string;
  updatedAt: string;
  entities: EpisodeSummaryEntity[];
  factsCount: number;
  processingDurationMs: number | null;
  error: string | null;
  retryCount: number;
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
  | "dead_letter";

export interface EpisodeSummaryEntity {
  id: string;
  name: string;
  type: EntityType;
}

// ─── Stats ──────────────────────────────────────────────────
export interface GraphStats {
  totalEntities: number;
  totalRelationships: number;
  totalEpisodes: number;
  entityTypeCounts: Record<EntityType, number>;
  topActivated: ActivationEntry[];
  topConnected: { id: string; name: string; degree: number }[];
  growthTimeline: { date: string; entities: number; episodes: number }[];
}

export interface ActivationEntry {
  entityId: string;
  name: string;
  entityType: EntityType;
  activation: number;
  neighborCount: number;  // aligned with MCP get_graph_state shape
}

// ─── WebSocket ──────────────────────────────────────────────
export type WsReadyState = "connecting" | "connected" | "reconnecting" | "disconnected";

// ─── UI State ───────────────────────────────────────────────
export type DashboardView =
  | "graph"
  | "timeline"
  | "feed"
  | "activation"
  | "stats";

export type GraphRenderMode = "3d" | "2d";

export interface TimeRange {
  start: string;  // ISO 8601
  end: string;    // ISO 8601
}
```

### 2.3 Store Slices

```typescript
// dashboard/src/store/index.ts

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ─── Graph Slice ────────────────────────────────────────────
interface GraphSlice {
  // State
  nodes: Map<string, GraphNode>;
  edges: Map<string, GraphEdge>;
  centerNodeId: string | null;  // current neighborhood center
  loadDepth: number;            // current expansion depth (1-3)
  isLoading: boolean;
  error: string | null;

  // Actions
  loadNeighborhood: (centerId: string, depth?: number) => Promise<void>;
  loadInitialGraph: () => Promise<void>;
  expandNode: (nodeId: string) => Promise<void>;
  mergeGraphDelta: (delta: GraphDelta) => void;
  updateNodeActivation: (nodeId: string, activation: number) => void;
  clear: () => void;
}

// ─── Selection Slice ────────────────────────────────────────
interface SelectionSlice {
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  selectedEdgeId: string | null;
  searchQuery: string;
  searchResults: GraphNode[];

  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  setSearchQuery: (query: string) => void;
  setSearchResults: (results: GraphNode[]) => void;
}

// ─── Time Slice ─────────────────────────────────────────────
interface TimeSlice {
  // null = "now" (live mode). When set, graph shows point-in-time state.
  timePosition: string | null; // ISO 8601
  timeRange: TimeRange | null; // visible window for timeline component
  isTimeScrubbing: boolean;

  setTimePosition: (iso: string | null) => void;
  setTimeRange: (range: TimeRange | null) => void;
  setTimeScrubbing: (v: boolean) => void;
}

// ─── Episode Slice ──────────────────────────────────────────
interface EpisodeSlice {
  episodes: Episode[];
  episodeCursor: string | null; // pagination cursor
  hasMore: boolean;
  isLoading: boolean;

  loadEpisodes: (cursor?: string) => Promise<void>;
  prependEpisode: (episode: Episode) => void;
  updateEpisodeStatus: (
    episodeId: string,
    status: EpisodeStatus,
    patch?: Partial<Episode>,
  ) => void;
}

// ─── Activation Slice ───────────────────────────────────────
interface ActivationSlice {
  topActivated: ActivationEntry[];
  activationHistory: Map<string, { timestamp: string; value: number }[]>;
  isMonitoring: boolean;

  setTopActivated: (entries: ActivationEntry[]) => void;
  pushActivationSnapshot: (
    entityId: string,
    timestamp: string,
    value: number,
  ) => void;
  setMonitoring: (v: boolean) => void;
}

// ─── Stats Slice ────────────────────────────────────────────
interface StatsSlice {
  stats: GraphStats | null;
  isLoading: boolean;
  loadStats: () => Promise<void>;
}

// ─── WebSocket Slice ────────────────────────────────────────
interface WebSocketSlice {
  readyState: WsReadyState;
  lastSeq: number;  // last processed server sequence number
  reconnectAttempt: number;
  missedEvents: number; // count of events dropped during reconnection

  setReadyState: (state: WsReadyState) => void;
  setLastSeq: (seq: number) => void;
  incrementReconnect: () => void;
  resetReconnect: () => void;
}

// ─── Preferences Slice (persisted to localStorage) ──────────
interface PreferencesSlice {
  currentView: DashboardView;
  renderMode: GraphRenderMode;
  showActivationHeatmap: boolean;
  showEdgeLabels: boolean;
  darkMode: boolean;
  graphMaxNodes: number; // max nodes to request (100-500)

  setView: (view: DashboardView) => void;
  setRenderMode: (mode: GraphRenderMode) => void;
  toggleActivationHeatmap: () => void;
  toggleEdgeLabels: () => void;
  toggleDarkMode: () => void;
  setGraphMaxNodes: (n: number) => void;
}

// ─── Combined Store ─────────────────────────────────────────
export type EngramStore = GraphSlice &
  SelectionSlice &
  TimeSlice &
  EpisodeSlice &
  ActivationSlice &
  StatsSlice &
  WebSocketSlice &
  PreferencesSlice;

export const useEngramStore = create<EngramStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Implementation of all slices.
        // Each slice is implemented in its own file and merged here.
        ...createGraphSlice(set, get),
        ...createSelectionSlice(set, get),
        ...createTimeSlice(set, get),
        ...createEpisodeSlice(set, get),
        ...createActivationSlice(set, get),
        ...createStatsSlice(set, get),
        ...createWebSocketSlice(set, get),
        ...createPreferencesSlice(set, get),
      })),
      {
        name: "engram-dashboard",
        // Only persist user preferences, not ephemeral data.
        partialize: (state) => ({
          currentView: state.currentView,
          renderMode: state.renderMode,
          showActivationHeatmap: state.showActivationHeatmap,
          showEdgeLabels: state.showEdgeLabels,
          darkMode: state.darkMode,
          graphMaxNodes: state.graphMaxNodes,
        }),
      },
    ),
    { name: "EngramStore" },
  ),
);
```

### 2.4 Graph Delta Merge

When WebSocket events arrive, the store must merge partial updates into the existing graph without a full re-fetch:

```typescript
// dashboard/src/store/graphSlice.ts

export interface GraphDelta {
  addedNodes: GraphNode[];
  updatedNodes: Partial<GraphNode & { id: string }>[];
  removedNodeIds: string[];
  addedEdges: GraphEdge[];
  updatedEdges: Partial<GraphEdge & { id: string }>[];
  removedEdgeIds: string[];
  activationChanges: { entityId: string; activation: number }[];
}

function mergeGraphDelta(state: EngramStore, delta: GraphDelta): void {
  // Add new nodes
  for (const node of delta.addedNodes) {
    state.nodes.set(node.id, node);
  }

  // Update existing nodes (partial merge)
  for (const patch of delta.updatedNodes) {
    const existing = state.nodes.get(patch.id!);
    if (existing) {
      state.nodes.set(patch.id!, { ...existing, ...patch });
    }
  }

  // Remove nodes
  for (const id of delta.removedNodeIds) {
    state.nodes.delete(id);
    // Also remove edges connected to this node
    for (const [edgeId, edge] of state.edges) {
      if (edge.source === id || edge.target === id) {
        state.edges.delete(edgeId);
      }
    }
  }

  // Add new edges
  for (const edge of delta.addedEdges) {
    state.edges.set(edge.id, edge);
  }

  // Update existing edges
  for (const patch of delta.updatedEdges) {
    const existing = state.edges.get(patch.id!);
    if (existing) {
      state.edges.set(patch.id!, { ...existing, ...patch });
    }
  }

  // Remove edges
  for (const id of delta.removedEdgeIds) {
    state.edges.delete(id);
  }

  // Update activation levels
  for (const { entityId, activation } of delta.activationChanges) {
    const node = state.nodes.get(entityId);
    if (node) {
      node.activationCurrent = activation;
    }
  }
}
```

---

## 3. REST API Endpoints

All endpoints require authentication (see `05_security_model.md`). The `group_id` is extracted from the `TenantContext` middleware -- it is never passed as a query parameter.

### 3.1 Subgraph Endpoint (Neighborhood-Based)

Replaces the original `GET /graph` (full graph) with a scalable neighborhood query.

```
GET /api/graph/neighborhood
```

**Query Parameters:**

| Param      | Type    | Default | Description |
|------------|---------|---------|-------------|
| `center`   | string  | *(none)* | Entity ID to center on. If omitted, server picks the highest-activated node. |
| `depth`    | integer | `2`     | BFS hop depth from center (1-3). |
| `max_nodes`| integer | `200`   | Maximum nodes to return. Server prunes by activation score when the neighborhood exceeds this limit. |
| `min_activation` | float | `0.0` | Only include nodes with `activation_current >= min_activation`. |

**Response:**

```json
{
  "center_id": "ent_abc123",
  "nodes": [
    {
      "id": "ent_abc123",
      "name": "ReadyCheck",
      "entity_type": "Project",
      "summary": "Meeting prep SaaS product",
      "activation_base": 0.65,
      "activation_current": 0.87,
      "access_count": 42,
      "last_accessed": "2026-02-27T10:30:00Z",
      "created_at": "2026-01-15T09:00:00Z",
      "updated_at": "2026-02-27T10:30:00Z"
    }
  ],
  "edges": [
    {
      "id": "rel_xyz789",
      "source": "ent_abc123",
      "target": "ent_def456",
      "predicate": "INTEGRATES",
      "weight": 0.8,
      "valid_from": "2026-02-20T14:00:00Z",
      "valid_to": null,
      "created_at": "2026-02-20T14:00:00Z"
    }
  ],
  "truncated": false,
  "total_in_neighborhood": 47
}
```

**Server-side algorithm:**

1. Start BFS from `center` up to `depth` hops.
2. Collect all reachable nodes (respecting `group_id` isolation).
3. If `|nodes| > max_nodes`, sort by `activation_current` descending, keep top `max_nodes`.
4. Include all edges where both endpoints are in the returned node set.
5. Set `truncated: true` if pruning occurred.

### 3.2 Entity Detail

```
GET /api/entities/{entity_id}
```

**Response:**

```json
{
  "id": "ent_abc123",
  "name": "ReadyCheck",
  "entity_type": "Project",
  "summary": "Meeting prep SaaS product with Stripe integration. 8-week launch target.",
  "activation_base": 0.65,
  "activation_current": 0.87,
  "access_count": 42,
  "last_accessed": "2026-02-27T10:30:00Z",
  "created_at": "2026-01-15T09:00:00Z",
  "updated_at": "2026-02-27T10:30:00Z",
  "facts": [
    {
      "id": "fact_001",
      "predicate": "INTEGRATES",
      "object": { "id": "ent_def456", "name": "Stripe" },
      "valid_from": "2026-02-20T14:00:00Z",
      "valid_to": null,
      "source_episode": "ep_01HXYZ9ABCDEF"
    },
    {
      "id": "fact_002",
      "predicate": "HAS_DEADLINE",
      "object": { "id": null, "name": "8-week launch (from 2026-01-15)" },
      "valid_from": "2026-01-15T09:00:00Z",
      "valid_to": null,
      "source_episode": "ep_01HABCDEF123"
    }
  ],
  "episodes": ["ep_01HXYZ9ABCDEF", "ep_01HABCDEF123"]
}
```

### 3.3 Entity Neighbors (Progressive Expand)

```
GET /api/entities/{entity_id}/neighbors
```

**Query Parameters:**

| Param      | Type    | Default | Description |
|------------|---------|---------|-------------|
| `depth`    | integer | `1`     | Hop depth (1-2). |
| `max_nodes`| integer | `50`    | Max neighbors to return, sorted by edge weight * activation. |

**Response:** Same shape as `GET /api/graph/neighborhood` response, centered on the given entity.

Used when the user clicks a node on the graph boundary to expand further.

### 3.4 Temporal Graph (Time-Scrubber)

```
GET /api/graph/at
```

**Query Parameters:**

| Param      | Type    | Default | Description |
|------------|---------|---------|-------------|
| `timestamp`| string  | *required* | ISO 8601 timestamp. Server evaluates which edges are valid at this point in time. |
| `center`   | string  | *(none)* | Optional center entity ID. If omitted, returns top-activated nodes that existed at that timestamp. |
| `depth`    | integer | `2`     | BFS depth. |
| `max_nodes`| integer | `200`   | Max nodes. |

**Server-side semantics:**

An edge is included if:
- `valid_from <= timestamp`
- `valid_to IS NULL OR valid_to > timestamp`
- `created_at <= timestamp` (the edge itself must have existed)

A node is included if:
- `created_at <= timestamp`
- It is reachable from the center within `depth` hops via valid edges at that timestamp.

Activation values are **not** replayed historically -- they show current activation. Historical activation replay would require storing activation snapshots per retrieval cycle, which is not in scope for v1. The time-scrubber is about **temporal graph topology** (what was true when), not historical activation.

**Response:** Same shape as `GET /api/graph/neighborhood`.

### 3.5 Entity Search

```
GET /api/entities/search
```

**Query Parameters:**

| Param      | Type    | Default | Description |
|------------|---------|---------|-------------|
| `q`        | string  | *required* | Search query (name substring or semantic). |
| `type`     | string  | *(all)* | Filter by entity type. |
| `limit`    | integer | `20`    | Max results. |

**Response:**

```json
{
  "results": [
    {
      "id": "ent_abc123",
      "name": "ReadyCheck",
      "entity_type": "Project",
      "summary": "Meeting prep SaaS product",
      "activation_current": 0.87,
      "match_type": "name"
    }
  ]
}
```

### 3.6 Episodes (Paginated Feed)

```
GET /api/episodes
```

**Query Parameters:**

| Param      | Type    | Default | Description |
|------------|---------|---------|-------------|
| `cursor`   | string  | *(none)* | Pagination cursor (episode_id). Returns episodes older than this cursor. |
| `limit`    | integer | `20`    | Page size (max 100). |
| `source`   | string  | *(all)* | Filter by source ("claude_desktop", "claude_code", "api"). |
| `status`   | string  | *(all)* | Filter by processing status. |

**Response:**

```json
{
  "episodes": [
    {
      "episode_id": "ep_01HXYZ9ABCDEF1234567890",
      "content": "I've been working on ReadyCheck...",
      "source": "claude_desktop",
      "status": "completed",
      "created_at": "2026-02-27T10:30:00Z",
      "updated_at": "2026-02-27T10:30:04Z",
      "entities": [
        { "id": "ent_abc123", "name": "ReadyCheck", "type": "Project" }
      ],
      "facts_count": 3,
      "processing_duration_ms": 4012,
      "error": null,
      "retry_count": 0
    }
  ],
  "next_cursor": "ep_01HABCDEF1234567890PREV",
  "has_more": true
}
```

### 3.7 Episode Status (Single)

```
GET /api/episodes/{episode_id}/status
```

Defined in `03_async_ingestion.md`. Returns the episode processing status.

### 3.8 Episode Admin Actions

```
POST /api/episodes/{episode_id}/requeue
```

Re-queues a dead-lettered episode for reprocessing. Returns `202 Accepted`.

```
DELETE /api/entities/{entity_id}
```

Soft-deletes an entity (sets `valid_to` on all edges). For GDPR hard-delete, use `DELETE /gdpr/erase` (see `05_security_model.md`).

```
PATCH /api/entities/{entity_id}
```

Update entity name or summary (for manual corrections from the dashboard).

**Request body:**

```json
{
  "name": "Corrected Name",
  "summary": "Updated summary text"
}
```

### 3.9 Dashboard Stats

```
GET /api/stats
```

**Response:**

```json
{
  "total_entities": 247,
  "total_relationships": 891,
  "total_episodes": 156,
  "entity_type_counts": {
    "Person": 23,
    "Organization": 15,
    "Project": 8,
    "Concept": 67,
    "Preference": 34,
    "Location": 12,
    "Event": 41,
    "Tool": 47
  },
  "top_activated": [
    { "entity_id": "ent_abc", "name": "ReadyCheck", "entity_type": "Project", "activation": 0.87, "neighbor_count": 12 }
  ],
  "top_connected": [
    { "id": "ent_user", "name": "User", "degree": 89 }
  ],
  "growth_timeline": [
    { "date": "2026-02-20", "entities": 201, "episodes": 130 },
    { "date": "2026-02-21", "entities": 215, "episodes": 138 }
  ]
}
```

### 3.10 MCP `get_graph_state` Shape

The MCP tool `get_graph_state` returns a summary for the MCP client (Claude Desktop). This is NOT the same as the dashboard subgraph endpoint -- it's a compact text-oriented summary.

```json
{
  "total_entities": 247,
  "total_relationships": 891,
  "total_episodes": 156,
  "top_activated": [
    { "name": "ReadyCheck", "type": "Project", "activation": 0.87 },
    { "name": "Stripe", "type": "Tool", "activation": 0.72 },
    { "name": "Python", "type": "Tool", "activation": 0.68 }
  ],
  "recent_entities": [
    { "name": "retatrutide", "type": "Concept", "created_at": "2026-02-27T10:30:00Z" }
  ],
  "graph_health": {
    "queue_depth": 0,
    "dead_letter_count": 0,
    "oldest_pending_episode": null
  }
}
```

The dashboard can also call `GET /api/stats` for the same data in a more structured form.

---

## 4. WebSocket Protocol

### 4.1 Connection

```
ws://localhost:8080/ws/dashboard
```

Auth: The WebSocket handshake includes the same `Authorization: Bearer <token>` header (or `engram_session` cookie in SaaS mode). The server validates auth before upgrading the connection. The `group_id` is extracted from the token -- no query parameter needed.

For self-hosted mode with auth disabled (`ENGRAM_AUTH_ENABLED=false`), the WebSocket accepts all connections and scopes to the default group.

### 4.2 Framing Protocol

Every message (both directions) is a JSON object with a standard envelope:

```typescript
// Server → Client
interface ServerMessage {
  seq: number;           // monotonically increasing per-connection sequence number
  type: string;          // event type (dot-namespaced)
  timestamp: string;     // ISO 8601, when the event was produced on the server
  payload: unknown;      // event-specific data
}

// Client → Server
interface ClientMessage {
  type: string;          // command type
  payload: unknown;
}
```

The `seq` number enables gap detection. If the client receives `seq: 42` followed by `seq: 44`, it knows `seq: 43` was lost (network hiccup, message drop) and should trigger a resync.

### 4.3 Server Event Types (server -> client)

#### `episode.queued`

Fired when a new episode enters the ingestion pipeline.

```json
{
  "seq": 1,
  "type": "episode.queued",
  "timestamp": "2026-02-27T10:30:00Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "source": "claude_desktop",
    "created_at": "2026-02-27T10:30:00Z"
  }
}
```

**Store update:** `episodeSlice.prependEpisode()` adds a skeleton episode with status "queued".

#### `episode.status_changed`

Fired on every pipeline stage transition.

```json
{
  "seq": 2,
  "type": "episode.status_changed",
  "timestamp": "2026-02-27T10:30:02Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "previous_status": "extracting",
    "status": "resolving",
    "updated_at": "2026-02-27T10:30:02Z"
  }
}
```

**Store update:** `episodeSlice.updateEpisodeStatus()`.

#### `episode.completed`

Fired when an episode finishes processing. Includes extracted entity summaries.

```json
{
  "seq": 5,
  "type": "episode.completed",
  "timestamp": "2026-02-27T10:30:04Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "entities": [
      { "id": "ent_abc123", "name": "ReadyCheck", "type": "Project" },
      { "id": "ent_def456", "name": "Stripe", "type": "Tool" }
    ],
    "relationships": [
      { "subject": "User", "predicate": "WORKS_ON", "object": "ReadyCheck" },
      { "subject": "ReadyCheck", "predicate": "INTEGRATES", "object": "Stripe" }
    ],
    "facts_count": 3,
    "processing_duration_ms": 4012,
    "updated_at": "2026-02-27T10:30:04Z"
  }
}
```

**Store update:** `episodeSlice.updateEpisodeStatus()` with full entity/fact data.

#### `episode.failed`

Fired when an episode is dead-lettered.

```json
{
  "seq": 6,
  "type": "episode.failed",
  "timestamp": "2026-02-27T10:35:00Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "error": "Claude API: 429 Too Many Requests",
    "retry_count": 5,
    "updated_at": "2026-02-27T10:35:00Z"
  }
}
```

**Store update:** `episodeSlice.updateEpisodeStatus("dead_letter", { error })`.

#### `graph.nodes_added`

Fired when new entities are created from an episode.

```json
{
  "seq": 7,
  "type": "graph.nodes_added",
  "timestamp": "2026-02-27T10:30:03Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "nodes": [
      {
        "id": "ent_new789",
        "name": "retatrutide",
        "entity_type": "Concept",
        "summary": "GLP-1/GIP/glucagon tri-agonist peptide",
        "activation_base": 0.1,
        "activation_current": 0.45,
        "access_count": 1,
        "last_accessed": "2026-02-27T10:30:03Z",
        "created_at": "2026-02-27T10:30:03Z",
        "updated_at": "2026-02-27T10:30:03Z"
      }
    ]
  }
}
```

**Store update:** `graphSlice.mergeGraphDelta({ addedNodes: payload.nodes })`.

#### `graph.edges_added`

Fired when new relationships are created.

```json
{
  "seq": 8,
  "type": "graph.edges_added",
  "timestamp": "2026-02-27T10:30:03Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "edges": [
      {
        "id": "rel_new001",
        "source": "ent_user",
        "target": "ent_new789",
        "predicate": "RESEARCHING",
        "weight": 0.6,
        "valid_from": "2026-02-27T10:30:03Z",
        "valid_to": null,
        "created_at": "2026-02-27T10:30:03Z"
      }
    ]
  }
}
```

**Store update:** `graphSlice.mergeGraphDelta({ addedEdges: payload.edges })`. Only adds edges where both endpoints exist in the current `nodes` map (the node may be outside the loaded neighborhood).

#### `graph.edges_invalidated`

Fired when an edge's `valid_to` is set (fact superseded by newer information).

```json
{
  "seq": 9,
  "type": "graph.edges_invalidated",
  "timestamp": "2026-02-27T10:30:03Z",
  "payload": {
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "edge_ids": ["rel_old002"],
    "valid_to": "2026-02-27T10:30:03Z"
  }
}
```

**Store update:** `graphSlice.mergeGraphDelta({ updatedEdges: [{ id, validTo }] })`.

#### `graph.nodes_updated`

Fired when existing entity attributes change (summary updated, merged with duplicate).

```json
{
  "seq": 10,
  "type": "graph.nodes_updated",
  "timestamp": "2026-02-27T10:30:03Z",
  "payload": {
    "nodes": [
      {
        "id": "ent_abc123",
        "summary": "Meeting prep SaaS product with Stripe integration. 8-week launch target from Jan 15.",
        "updated_at": "2026-02-27T10:30:03Z"
      }
    ]
  }
}
```

**Store update:** `graphSlice.mergeGraphDelta({ updatedNodes: payload.nodes })`.

#### `activation.updated`

Fired after activation state changes (post-ingestion, post-retrieval, or post-spreading).

```json
{
  "seq": 11,
  "type": "activation.updated",
  "timestamp": "2026-02-27T10:30:04Z",
  "payload": {
    "changes": [
      { "entity_id": "ent_abc123", "name": "ReadyCheck", "activation": 0.87 },
      { "entity_id": "ent_def456", "name": "Stripe", "activation": 0.72 },
      { "entity_id": "ent_new789", "name": "retatrutide", "activation": 0.45 }
    ],
    "trigger": "ingestion",
    "source_episode_id": "ep_01HXYZ9ABCDEF1234567890"
  }
}
```

**Store update:** Updates `graphSlice` node activation values. Also pushes to `activationSlice.pushActivationSnapshot()` for the Activation Monitor chart. The `trigger` field ("ingestion" | "retrieval" | "decay") tells the Activation Monitor view what caused the change.

#### `graph.node_removed`

Fired when an entity is deleted (via dashboard or GDPR erasure).

```json
{
  "seq": 12,
  "type": "graph.node_removed",
  "timestamp": "2026-02-27T11:00:00Z",
  "payload": {
    "entity_id": "ent_old999",
    "reason": "user_delete"
  }
}
```

**Store update:** `graphSlice.mergeGraphDelta({ removedNodeIds: [payload.entity_id] })`.

### 4.4 Client Command Types (client -> server)

#### `subscribe.activation_monitor`

Start receiving high-frequency activation snapshots (for the Activation Monitor view). Without this, `activation.updated` events are only sent on ingestion/retrieval triggers, not on periodic intervals.

```json
{
  "type": "subscribe.activation_monitor",
  "payload": {
    "interval_ms": 2000,
    "top_n": 20
  }
}
```

Server starts sending `activation.updated` events every `interval_ms` with the top-N activated entities.

#### `unsubscribe.activation_monitor`

Stop the high-frequency activation feed.

```json
{
  "type": "unsubscribe.activation_monitor",
  "payload": {}
}
```

#### `ping`

Client heartbeat. Server responds with `pong`.

```json
{ "type": "ping", "payload": {} }
```

Server responds:

```json
{
  "seq": 13,
  "type": "pong",
  "timestamp": "2026-02-27T10:31:00Z",
  "payload": { "server_time": "2026-02-27T10:31:00Z" }
}
```

### 4.5 Event Ordering: Ingestion Sequence

When an episode is ingested, the server sends events in this order:

```
1. episode.queued              (remember() called)
2. episode.status_changed      (extracting)
3. episode.status_changed      (resolving)
4. episode.status_changed      (writing)
5. graph.nodes_added           (new entities created during write)
6. graph.edges_added           (new relationships created during write)
7. graph.edges_invalidated     (old facts superseded, if any)
8. graph.nodes_updated         (entity summaries changed, if any)
9. episode.status_changed      (embedding)
10. episode.status_changed     (activating)
11. activation.updated         (activation state changed)
12. episode.completed          (terminal)
```

Not all events fire for every episode. If no new nodes are created (all entities already existed), `graph.nodes_added` is skipped. Events 5-8 and 11 are only relevant to the dashboard if the affected nodes are in the currently loaded neighborhood.

---

## 5. Reconnection Protocol

### 5.1 Strategy

The client tracks the last `seq` number it received. On reconnect, it attempts sequence-based resync first, falling back to a full REST refetch if the gap is too large.

### 5.2 Connection Lifecycle

```
                    Initial Connect
                          |
                    ┌─────v──────┐
                    │ CONNECTING  │
                    └─────┬──────┘
                          |
                    ws.onopen
                          |
                    ┌─────v──────┐
               ┌───>│ CONNECTED   │<──────────────────────┐
               │    └─────┬──────┘                        │
               │          |                                │
               │    ws.onclose / ws.onerror                │
               │          |                                │
               │    ┌─────v──────────┐                     │
               │    │ RECONNECTING   │                     │
               │    └─────┬──────────┘                     │
               │          |                                │
               │    exponential backoff                    │
               │    (1s, 2s, 4s, 8s, 16s, 30s max)       │
               │          |                                │
               │    ┌─────v──────┐                         │
               │    │ CONNECTING  │─── success ────────────┘
               │    └─────┬──────┘
               │          |
               │    5 consecutive failures
               │          |
               │    ┌─────v──────────┐
               └────│ DISCONNECTED   │ (user sees banner: "Connection lost")
                    └────────────────┘
                          |
                    user clicks "Reconnect"
                          |
                    reset attempt counter, goto CONNECTING
```

### 5.3 Resync on Reconnect

```typescript
// dashboard/src/lib/ws.ts

async function onReconnect(ws: WebSocket, lastSeq: number): Promise<void> {
  // Send resync request with last known sequence number
  ws.send(JSON.stringify({
    type: "resync",
    payload: { last_seq: lastSeq },
  }));
}
```

**Server behavior on `resync` request:**

1. Server maintains a per-connection event buffer of the last 1000 events (ring buffer in memory).
2. If `last_seq` is within the buffer, replay all events from `last_seq + 1` to current. Client processes them in order, updating the store.
3. If `last_seq` is too old (outside the buffer), server responds with:

```json
{
  "seq": 0,
  "type": "resync.full_required",
  "timestamp": "2026-02-27T10:45:00Z",
  "payload": {
    "reason": "gap_too_large",
    "missed_events": 1247,
    "current_seq": 2500
  }
}
```

4. On `resync.full_required`, the client:
   a. Calls `GET /api/graph/neighborhood` to reload the current viewport.
   b. Calls `GET /api/episodes?limit=20` to reload the feed.
   c. Calls `GET /api/stats` to reload stats.
   d. Sets `lastSeq` to the server's `current_seq` from the resync response.
   e. Resumes normal WebSocket event processing.

### 5.4 Backoff Schedule

```typescript
const BACKOFF_BASE_MS = 1000;
const BACKOFF_MAX_MS = 30_000;
const MAX_RECONNECT_ATTEMPTS = 5; // before showing "Disconnected" banner

function getBackoffDelay(attempt: number): number {
  const delay = Math.min(
    BACKOFF_BASE_MS * Math.pow(2, attempt),
    BACKOFF_MAX_MS,
  );
  // Add jitter: +/- 20%
  const jitter = delay * 0.2 * (Math.random() * 2 - 1);
  return Math.round(delay + jitter);
}
```

### 5.5 WebSocket Hook

```typescript
// dashboard/src/hooks/useWebSocket.ts

import { useEffect, useRef } from "react";
import { useEngramStore } from "../store";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const setReadyState = useEngramStore((s) => s.setReadyState);
  const setLastSeq = useEngramStore((s) => s.setLastSeq);
  const lastSeq = useEngramStore((s) => s.lastSeq);
  const incrementReconnect = useEngramStore((s) => s.incrementReconnect);
  const resetReconnect = useEngramStore((s) => s.resetReconnect);
  const reconnectAttempt = useEngramStore((s) => s.reconnectAttempt);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      setReadyState("connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/dashboard`);

      ws.onopen = () => {
        setReadyState("connected");
        resetReconnect();

        // Resync if we have a previous sequence number
        if (lastSeq > 0) {
          ws.send(JSON.stringify({
            type: "resync",
            payload: { last_seq: lastSeq },
          }));
        }
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as ServerMessage;
        setLastSeq(msg.seq);
        handleServerMessage(msg);
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (reconnectAttempt < MAX_RECONNECT_ATTEMPTS) {
          setReadyState("reconnecting");
          incrementReconnect();
          reconnectTimer = setTimeout(connect, getBackoffDelay(reconnectAttempt));
        } else {
          setReadyState("disconnected");
        }
      };

      ws.onerror = () => {
        ws.close(); // triggers onclose -> reconnect
      };

      wsRef.current = ws;
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);  // mount once
}
```

---

## 6. 2D Fallback Mode

### 6.1 When to Use

- **Accessibility.** Screen readers cannot interpret WebGL 3D scenes. 2D mode with proper ARIA labels is accessible.
- **Low-end devices.** Mobile browsers, old laptops, devices without GPU acceleration.
- **User preference.** Some users simply prefer a flat layout.

### 6.2 Implementation

Both modes use the `react-force-graph` library:

| Mode | Package | Renderer | Camera |
|------|---------|----------|--------|
| 3D   | `react-force-graph-3d` | Three.js (WebGL) | Orbit controls (rotate, zoom, pan) |
| 2D   | `react-force-graph-2d` | Canvas 2D | Pan + zoom |

The graph data, store slices, and event handling are identical. Only the render component differs.

```typescript
// dashboard/src/components/GraphExplorer.tsx

import ForceGraph3D from "react-force-graph-3d";
import ForceGraph2D from "react-force-graph-2d";
import { useEngramStore } from "../store";

export function GraphExplorer() {
  const renderMode = useEngramStore((s) => s.renderMode);
  const nodes = useEngramStore((s) => s.nodes);
  const edges = useEngramStore((s) => s.edges);

  const graphData = useMemo(() => ({
    nodes: Array.from(nodes.values()),
    links: Array.from(edges.values()).map((e) => ({
      source: e.source,
      target: e.target,
      ...e,
    })),
  }), [nodes, edges]);

  const ForceGraph = renderMode === "3d" ? ForceGraph3D : ForceGraph2D;

  return (
    <ForceGraph
      graphData={graphData}
      nodeLabel={(node: GraphNode) => node.name}
      nodeVal={(node: GraphNode) => nodeSize(node.activationCurrent)}
      nodeColor={(node: GraphNode) => entityTypeColor(node.entityType)}
      linkWidth={(link: GraphEdge) => link.weight * 3}
      linkLabel={(link: GraphEdge) => link.predicate}
      onNodeClick={(node: GraphNode) => selectNode(node.id)}
      onNodeHover={(node: GraphNode | null) => hoverNode(node?.id ?? null)}
      // ... shared config
    />
  );
}
```

### 6.3 Auto-Detection

On first load, detect WebGL support:

```typescript
function detectWebGLSupport(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      canvas.getContext("webgl2") || canvas.getContext("webgl")
    );
  } catch {
    return false;
  }
}

// In PreferencesSlice initialization:
const defaultRenderMode: GraphRenderMode = detectWebGLSupport() ? "3d" : "2d";
```

---

## 7. Component Architecture

### 7.1 Component Tree

```
App
├── AuthGate                          # Checks auth, shows login if SaaS mode
│   └── DashboardShell                # Layout: sidebar + main content
│       ├── Sidebar
│       │   ├── ViewSwitcher          # graph | timeline | feed | activation | stats
│       │   ├── SearchBar             # Entity search with typeahead
│       │   └── ConnectionStatus      # WebSocket status indicator
│       ├── TopBar
│       │   ├── GraphControls         # 2D/3D toggle, heatmap toggle, edge labels
│       │   └── TimeScrubber          # Time-scrubber slider (visible in graph + timeline views)
│       └── MainContent
│           ├── GraphExplorer         # 3D/2D force graph (view: "graph")
│           │   ├── NodeTooltip       # Hover: name, type, activation, last accessed
│           │   └── NodeDetailPanel   # Click: facts, episodes, neighbors, actions
│           ├── TimelineView          # Horizontal timeline (view: "timeline")
│           │   ├── EntityTimeline    # Entity creation/update markers
│           │   ├── FactValidityBars  # Temporal validity ranges
│           │   └── EpisodeMarkers    # Episode source labels
│           ├── MemoryFeed           # Reverse-chrono episode list (view: "feed")
│           │   ├── EpisodeCard       # Single episode with extracted entities/facts
│           │   └── EpisodeActions    # Requeue, delete, expand
│           ├── ActivationMonitor    # Live activation view (view: "activation")
│           │   ├── ActivationChart   # Top-20 bar chart (Recharts)
│           │   ├── SpreadingViz      # Spreading activation animation overlay
│           │   └── DecayCurves       # Decay over time line chart
│           └── StatsPanel           # Aggregate stats (view: "stats")
│               ├── CountCards        # Total entities, relationships, episodes
│               ├── GrowthChart       # Growth over time (Recharts)
│               ├── TypeDistribution  # Entity type pie chart
│               └── TopConnected      # Hub nodes leaderboard
```

### 7.2 Component -> Store Slice Mapping

| Component | Reads From | Writes To |
|-----------|-----------|-----------|
| `GraphExplorer` | `graphSlice.nodes`, `graphSlice.edges`, `selectionSlice.hoveredNodeId`, `preferencesSlice.renderMode`, `preferencesSlice.showActivationHeatmap`, `preferencesSlice.showEdgeLabels` | `selectionSlice.selectNode`, `selectionSlice.hoverNode`, `graphSlice.expandNode` |
| `NodeTooltip` | `selectionSlice.hoveredNodeId`, `graphSlice.nodes` | *(none)* |
| `NodeDetailPanel` | `selectionSlice.selectedNodeId`, `graphSlice.nodes`, `graphSlice.edges` | `graphSlice.expandNode`, `selectionSlice.selectNode` |
| `SearchBar` | `selectionSlice.searchQuery`, `selectionSlice.searchResults` | `selectionSlice.setSearchQuery`, `selectionSlice.setSearchResults`, `graphSlice.loadNeighborhood` |
| `TimeScrubber` | `timeSlice.timePosition`, `timeSlice.timeRange` | `timeSlice.setTimePosition`, `timeSlice.setTimeScrubbing`, `graphSlice.loadNeighborhood` (re-fetch at timestamp) |
| `MemoryFeed` | `episodeSlice.episodes`, `episodeSlice.hasMore`, `episodeSlice.isLoading` | `episodeSlice.loadEpisodes` |
| `EpisodeCard` | `episodeSlice.episodes[i]` | `selectionSlice.selectNode` (click entity link) |
| `ActivationMonitor` | `activationSlice.topActivated`, `activationSlice.activationHistory` | `activationSlice.setMonitoring` (toggles WebSocket subscription) |
| `StatsPanel` | `statsSlice.stats`, `statsSlice.isLoading` | `statsSlice.loadStats` |
| `ConnectionStatus` | `wsSlice.readyState`, `wsSlice.reconnectAttempt` | *(none)* |
| `ViewSwitcher` | `preferencesSlice.currentView` | `preferencesSlice.setView` |
| `GraphControls` | `preferencesSlice.*` | `preferencesSlice.setRenderMode`, `preferencesSlice.toggleActivationHeatmap`, etc. |

### 7.3 Data Loading Patterns

**Initial load (page open):**

1. `useWebSocket()` hook connects WebSocket.
2. `graphSlice.loadInitialGraph()` calls `GET /api/graph/neighborhood` (no center = server picks top-activated node).
3. `episodeSlice.loadEpisodes()` fetches first page.
4. `statsSlice.loadStats()` fetches stats.
5. Steps 2-4 run in parallel via `Promise.all`.

**Node click (expand):**

1. `selectionSlice.selectNode(nodeId)` sets selected node.
2. If the node is on the graph boundary (not all neighbors loaded), `graphSlice.expandNode(nodeId)` calls `GET /api/entities/{id}/neighbors`.
3. Response merged into store via `mergeGraphDelta`.

**Time scrub:**

1. `timeSlice.setTimePosition(timestamp)` sets the position.
2. Debounced (300ms) call to `GET /api/graph/at?timestamp={iso}&center={currentCenter}`.
3. Response replaces graph data (not merged -- it's a different point-in-time view).
4. Setting `timePosition = null` returns to live mode and re-fetches current neighborhood.

**Search:**

1. User types in `SearchBar`. Debounced (250ms) call to `GET /api/entities/search?q={query}`.
2. Results populate `selectionSlice.searchResults`.
3. User clicks a result: `graphSlice.loadNeighborhood(entityId)` re-centers the graph on that entity.

---

## 8. Activation Heatmap

### 8.1 Visual Encoding

Nodes are colored by a gradient mapped to their `activationCurrent` value:

```typescript
// dashboard/src/lib/colors.ts

/**
 * Activation heatmap color scale.
 * Maps activation (0.0 - 1.0) to a color from cool blue to warm red.
 */
export function activationColor(activation: number): string {
  // HSL interpolation: 240 (blue) -> 0 (red)
  const hue = 240 * (1 - activation);
  const saturation = 70 + activation * 20; // 70-90%
  const lightness = 50 + (1 - activation) * 15; // dim cold nodes slightly
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Entity type colors (when heatmap is disabled).
 */
export const ENTITY_TYPE_COLORS: Record<EntityType, string> = {
  Person:       "#6366f1", // indigo
  Organization: "#8b5cf6", // violet
  Project:      "#06b6d4", // cyan
  Concept:      "#10b981", // emerald
  Preference:   "#f59e0b", // amber
  Location:     "#ef4444", // red
  Event:        "#ec4899", // pink
  Tool:         "#3b82f6", // blue
};
```

When `showActivationHeatmap` is true, node colors use `activationColor()`. When false, they use `ENTITY_TYPE_COLORS`.

### 8.2 Node Sizing

Node size scales with activation, with a minimum size so dormant nodes remain visible:

```typescript
function nodeSize(activation: number): number {
  const MIN_SIZE = 3;
  const MAX_SIZE = 20;
  return MIN_SIZE + activation * (MAX_SIZE - MIN_SIZE);
}
```

---

## 9. CORS & Proxy Config for Local Development

### 9.1 Vite Dev Server Proxy

During development, the React dashboard runs on Vite's dev server (port 5173) and the FastAPI backend runs on port 8080. To avoid CORS issues in development, Vite proxies API and WebSocket requests:

```typescript
// dashboard/vite.config.ts

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
      },
    },
  },
});
```

With this config, the frontend code uses relative URLs (`/api/graph/neighborhood`, `/ws/dashboard`) and the proxy forwards to FastAPI. No CORS configuration is needed for local dev.

### 9.2 FastAPI CORS (Production / Docker)

In production (or Docker Compose), the dashboard is served as static files by Nginx or the FastAPI app itself. CORS is configured per `05_security_model.md`:

```yaml
# config.yaml
cors:
  allowed_origins:
    - "http://localhost:3000"   # Docker dashboard
    - "http://localhost:5173"   # Vite dev server
```

### 9.3 API Client

```typescript
// dashboard/src/api/client.ts

const BASE_URL = "";  // relative (proxied in dev, same-origin in prod)

interface FetchOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
}

async function apiFetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
      // In SaaS mode, the JWT cookie is sent automatically.
      // In self-hosted mode with auth, the token is stored in localStorage
      // and attached here:
      ...(getStoredToken() ? { Authorization: `Bearer ${getStoredToken()}` } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
    credentials: "include", // send cookies for SaaS JWT
    signal: options.signal,
  });

  if (!response.ok) {
    if (response.status === 401) {
      // Token expired or invalid -- redirect to login
      useEngramStore.getState().setView("graph"); // reset view
      window.location.href = "/login";
      throw new Error("Unauthorized");
    }
    const errorBody = await response.json().catch(() => ({}));
    throw new ApiError(response.status, errorBody.detail ?? "Request failed");
  }

  return response.json();
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ─── Typed API methods ──────────────────────────────────────

export const api = {
  getNeighborhood: (params: {
    center?: string;
    depth?: number;
    maxNodes?: number;
    minActivation?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params.center) qs.set("center", params.center);
    if (params.depth) qs.set("depth", String(params.depth));
    if (params.maxNodes) qs.set("max_nodes", String(params.maxNodes));
    if (params.minActivation) qs.set("min_activation", String(params.minActivation));
    return apiFetch<NeighborhoodResponse>(`/api/graph/neighborhood?${qs}`);
  },

  getGraphAt: (params: {
    timestamp: string;
    center?: string;
    depth?: number;
    maxNodes?: number;
  }) => {
    const qs = new URLSearchParams({ timestamp: params.timestamp });
    if (params.center) qs.set("center", params.center);
    if (params.depth) qs.set("depth", String(params.depth));
    if (params.maxNodes) qs.set("max_nodes", String(params.maxNodes));
    return apiFetch<NeighborhoodResponse>(`/api/graph/at?${qs}`);
  },

  getEntity: (id: string) =>
    apiFetch<EntityDetail>(`/api/entities/${id}`),

  getNeighbors: (id: string, params?: { depth?: number; maxNodes?: number }) => {
    const qs = new URLSearchParams();
    if (params?.depth) qs.set("depth", String(params.depth));
    if (params?.maxNodes) qs.set("max_nodes", String(params.maxNodes));
    return apiFetch<NeighborhoodResponse>(`/api/entities/${id}/neighbors?${qs}`);
  },

  searchEntities: (params: { q: string; type?: string; limit?: number }) => {
    const qs = new URLSearchParams({ q: params.q });
    if (params.type) qs.set("type", params.type);
    if (params.limit) qs.set("limit", String(params.limit));
    return apiFetch<SearchResponse>(`/api/entities/search?${qs}`);
  },

  getEpisodes: (params?: { cursor?: string; limit?: number; source?: string; status?: string }) => {
    const qs = new URLSearchParams();
    if (params?.cursor) qs.set("cursor", params.cursor);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.source) qs.set("source", params.source);
    if (params?.status) qs.set("status", params.status);
    return apiFetch<EpisodesResponse>(`/api/episodes?${qs}`);
  },

  getEpisodeStatus: (id: string) =>
    apiFetch<Episode>(`/api/episodes/${id}/status`),

  getStats: () =>
    apiFetch<GraphStats>(`/api/stats`),

  requeueEpisode: (id: string) =>
    apiFetch<void>(`/api/episodes/${id}/requeue`, { method: "POST" }),

  deleteEntity: (id: string) =>
    apiFetch<void>(`/api/entities/${id}`, { method: "DELETE" }),

  updateEntity: (id: string, body: { name?: string; summary?: string }) =>
    apiFetch<EntityDetail>(`/api/entities/${id}`, { method: "PATCH", body }),
};
```

---

## 10. Dashboard Auth Flow

### 10.1 Self-Hosted Mode (Bearer Token)

In self-hosted mode, there is no login screen. Auth is handled via configuration:

1. User sets `ENGRAM_AUTH_SECRET` in `.env`.
2. Dashboard determines auth requirements via `GET /api/config/dashboard` (no auth required), which returns `{ auth_required: bool }`. If `false` (default for self-hosted with `ENGRAM_AUTH_ENABLED=false`), no token needed.
3. If `auth_required: true` and no token is stored in localStorage, the dashboard shows a one-time token entry modal: "Enter your Engram auth token". The token is stored to localStorage and never appears in the URL.
4. Subsequent requests include `Authorization: Bearer <token>` header.

**Security note:** The token is never passed via URL query parameters. Query parameters are logged in browser history, server access logs, and the `Referer` header, making them unsuitable for bearer tokens. The token entry modal pattern follows the approach used by Grafana, Portainer, and similar self-hosted tools.

When `ENGRAM_AUTH_ENABLED=false` (default for development), the dashboard works with zero auth configuration.

### 10.2 SaaS Mode (JWT)

1. User navigates to dashboard. `AuthGate` component checks for valid `engram_session` cookie.
2. If no cookie (or expired), redirect to `/login`.
3. Login page: email + password form, posts to `POST /auth/login`.
4. Server returns `Set-Cookie: engram_session=<jwt>` (httpOnly, secure, sameSite=strict).
5. All subsequent API requests include the cookie automatically (`credentials: "include"`).
6. JWT has 15-minute TTL. A background refresh runs every 10 minutes via `POST /auth/refresh` (sends the refresh token from a separate cookie).
7. On 401 from any API call, redirect to `/login`.

```typescript
// dashboard/src/components/AuthGate.tsx

import { useEffect, useState } from "react";
import { api } from "../api/client";

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [authState, setAuthState] = useState<"checking" | "authenticated" | "login_required">("checking");

  useEffect(() => {
    api.getStats()
      .then(() => setAuthState("authenticated"))
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          setAuthState("login_required");
        } else {
          // Server unreachable or other error -- still show dashboard
          // with error state, not login page
          setAuthState("authenticated");
        }
      });
  }, []);

  if (authState === "checking") return <LoadingSpinner />;
  if (authState === "login_required") return <LoginPage />;
  return <>{children}</>;
}
```

---

## 11. Error States & Loading

### 11.1 Loading States

| Component | Loading Trigger | Indicator |
|-----------|----------------|-----------|
| `GraphExplorer` | `loadNeighborhood`, `loadInitialGraph` | Skeleton graph with pulsing placeholder nodes |
| `MemoryFeed` | `loadEpisodes` | Skeleton cards |
| `StatsPanel` | `loadStats` | Skeleton count cards |
| `NodeDetailPanel` | Fetching entity detail | Inline spinner |
| `SearchBar` | Debounced search | Spinner in input |

### 11.2 Error States

| Error | User-Facing Message | Recovery |
|-------|-------------------|----------|
| API unreachable | "Cannot reach Engram server. Is it running?" | Retry button |
| 401 Unauthorized | Redirect to login (SaaS) or "Invalid token" banner (self-hosted) | Re-enter token |
| 429 Rate Limited | "Too many requests. Retrying in {n}s..." | Auto-retry with backoff |
| WebSocket disconnect | Yellow banner: "Live updates paused. Reconnecting..." | Auto-reconnect |
| WebSocket disconnected (5 failures) | Red banner: "Connection lost." | Manual "Reconnect" button |
| Graph empty | "No memories yet. Start a conversation with Claude." | Link to setup guide |

---

## 12. Tech Stack Summary

| Concern | Technology | Version |
|---------|-----------|---------|
| Framework | React | 18+ |
| Language | TypeScript | 5.x |
| Build | Vite | 5.x |
| State | Zustand | 4.x |
| Immutable updates | immer (via zustand/middleware/immer) | 10.x |
| Styling | Tailwind CSS | 3.x |
| Graph (3D) | react-force-graph-3d | 1.x |
| Graph (2D) | react-force-graph-2d | 1.x |
| Charts | Recharts | 2.x |
| Icons | Lucide React | latest |
| HTTP | fetch (native) | - |
| WebSocket | WebSocket (native) | - |

No additional HTTP client library (axios, ky) is needed. The native `fetch` API with the typed wrapper in `api/client.ts` is sufficient.

---

## 13. File Structure

```
dashboard/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.ts
├── Dockerfile
├── index.html
├── public/
│   └── favicon.svg
└── src/
    ├── main.tsx                    # React root, providers
    ├── App.tsx                     # AuthGate + DashboardShell
    ├── store/
    │   ├── index.ts                # Combined store creation
    │   ├── types.ts                # All TypeScript types
    │   ├── graphSlice.ts           # Graph nodes/edges state
    │   ├── selectionSlice.ts       # Selected/hovered node
    │   ├── timeSlice.ts            # Time-scrubber state
    │   ├── episodeSlice.ts         # Episode feed state
    │   ├── activationSlice.ts      # Activation monitor state
    │   ├── statsSlice.ts           # Stats state
    │   ├── wsSlice.ts              # WebSocket connection state
    │   └── preferencesSlice.ts     # Persisted user preferences
    ├── api/
    │   └── client.ts               # Typed REST client
    ├── hooks/
    │   ├── useWebSocket.ts         # WebSocket connection + event routing
    │   └── useDebounce.ts          # Debounce utility
    ├── lib/
    │   ├── colors.ts               # Activation heatmap + entity type colors
    │   ├── webgl.ts                # WebGL detection
    │   └── format.ts               # Date/number formatting
    ├── components/
    │   ├── AuthGate.tsx
    │   ├── DashboardShell.tsx
    │   ├── Sidebar.tsx
    │   ├── TopBar.tsx
    │   ├── ConnectionStatus.tsx
    │   ├── SearchBar.tsx
    │   ├── TimeScrubber.tsx
    │   ├── GraphExplorer.tsx
    │   ├── GraphControls.tsx
    │   ├── NodeTooltip.tsx
    │   ├── NodeDetailPanel.tsx
    │   ├── TimelineView.tsx
    │   ├── MemoryFeed.tsx
    │   ├── EpisodeCard.tsx
    │   ├── ActivationMonitor.tsx
    │   ├── StatsPanel.tsx
    │   └── LoginPage.tsx
    └── styles/
        └── globals.css             # Tailwind imports + custom vars
```

---

## 14. Implementation Priority

| Priority | Component | Week | Notes |
|----------|-----------|------|-------|
| P0 | Zustand store + types | 5 | Foundation for all components |
| P0 | REST API endpoints (neighborhood, entity, search) | 5 | Backend must expose these |
| P0 | API client (`client.ts`) | 5 | |
| P0 | GraphExplorer (3D) | 5 | The "screenshot moment" |
| P0 | WebSocket connection + event routing | 5 | Live updates |
| P1 | NodeTooltip + NodeDetailPanel | 5 | Interactivity |
| P1 | SearchBar | 5 | |
| P1 | 2D fallback mode | 5 | |
| P1 | MemoryFeed + EpisodeCard | 6 | |
| P1 | TimelineView + TimeScrubber | 6 | Temporal exploration |
| P1 | StatsPanel | 6 | |
| P2 | ActivationMonitor | 7 | Live activation visualization |
| P2 | Activation heatmap overlay | 7 | |
| P2 | Reconnection protocol (full resync) | 7 | Robustness polish |
| P2 | AuthGate + LoginPage (SaaS) | Month 3 | Not needed for self-hosted |
| P2 | Dark mode | 7 | |

---

## 15. Cross-Document Alignment

### 15.1 Ingestion Pipeline (`03_async_ingestion.md`)

The WebSocket events defined here (Section 4.3) are a superset of those defined in `03_async_ingestion.md`. The ingestion doc defines the episode-lifecycle events (`episode.queued`, `episode.status_changed`, `episode.completed`, `episode.failed`, `graph.updated`). This document splits `graph.updated` into granular events (`graph.nodes_added`, `graph.edges_added`, `graph.edges_invalidated`, `graph.nodes_updated`, `activation.updated`) so the frontend can perform surgical store updates instead of full re-renders.

The ingestion pipeline worker is responsible for publishing all events to the WebSocket manager. The event sequence (Section 4.5) must be followed to ensure the dashboard store remains consistent.

### 15.2 Auth Layer (`05_security_model.md`)

- Dashboard auth uses the JWT cookie flow (SaaS) or bearer token (self-hosted) as defined in the security doc.
- CORS origins include `http://localhost:5173` (Vite dev) and `http://localhost:3000` (Docker).
- WebSocket connections are authenticated via the same `TenantContextMiddleware`.
- Rate limits for WebSocket: max 5 connections per tenant (Section 10.2 of security doc).

### 15.3 MCP Tools

The `get_graph_state` MCP tool response shape (Section 3.10) should be aligned with whatever schema the MCP agent defines. The dashboard's `GET /api/stats` endpoint returns the same data in a structured format suitable for rendering.

### 15.4 Config Schema (`01_config_and_indexes.md`)

No new config fields are required for the frontend itself. The dashboard is a static build. The backend config fields that affect the frontend:
- `cors.allowed_origins` -- must include the dashboard URL.
- `auth.mode` -- determines whether the dashboard shows a login page.
- `auth.bearer_token` / `auth.jwt.*` -- used by the auth middleware.
