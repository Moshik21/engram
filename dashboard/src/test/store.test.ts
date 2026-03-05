import { describe, it, expect, vi, beforeEach } from "vitest";
import { createStore } from "zustand";
import { immer } from "zustand/middleware/immer";
import type { EngramStore, GraphNode, GraphEdge, SearchResult, Episode } from "../store/types";
import { createGraphSlice } from "../store/graphSlice";
import { createSelectionSlice } from "../store/selectionSlice";
import { createPreferencesSlice } from "../store/preferencesSlice";
import { createTimeSlice } from "../store/timeSlice";
import { createEpisodeSlice } from "../store/episodeSlice";
import { createStatsSlice } from "../store/statsSlice";
import { createWsSlice } from "../store/wsSlice";
import { createActivationSlice } from "../store/activationSlice";
import { createConsolidationSlice } from "../store/consolidationSlice";
import { createKnowledgeSlice } from "../store/knowledgeSlice";
import { createConversationSlice } from "../store/conversationSlice";

vi.mock("../api/client", () => ({
  api: {
    getNeighborhood: vi.fn(),
    getNeighbors: vi.fn(),
    searchEntities: vi.fn(),
    getEntity: vi.fn(),
    getStats: vi.fn(),
    getEpisodes: vi.fn(),
    getGraphAt: vi.fn(),
    updateEntity: vi.fn(),
    deleteEntity: vi.fn(),
    getActivationSnapshot: vi.fn(),
    getActivationCurve: vi.fn(),
    getConsolidationStatus: vi.fn(),
    getConsolidationHistory: vi.fn(),
    getConsolidationCycle: vi.fn(),
    triggerConsolidation: vi.fn(),
    recall: vi.fn(),
    searchFacts: vi.fn(),
    getKnowledgeContext: vi.fn(),
    observe: vi.fn(),
    remember: vi.fn(),
    forget: vi.fn(),
    listConversations: vi.fn(),
    getConversationMessages: vi.fn(),
    appendConversationMessages: vi.fn(),
  },
}));

import { api } from "../api/client";

const mockedApi = vi.mocked(api);

function createTestStore() {
  return createStore<EngramStore>()(
    immer((...a) => ({
      ...createGraphSlice(...a),
      ...createSelectionSlice(...a),
      ...createPreferencesSlice(...a),
      ...createTimeSlice(...a),
      ...createEpisodeSlice(...a),
      ...createStatsSlice(...a),
      ...createWsSlice(...a),
      ...createActivationSlice(...a),
      ...createConsolidationSlice(...a),
      ...createKnowledgeSlice(...a),
      ...createConversationSlice(...(a as Parameters<typeof createConversationSlice>)),
    })),
  );
}

function makeNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id: "n1",
    name: "Test Node",
    entityType: "Person",
    summary: null,
    activationCurrent: 0.5,
    accessCount: 3,
    lastAccessed: "2024-01-01T00:00:00Z",
    createdAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeEdge(overrides: Partial<GraphEdge> = {}): GraphEdge {
  return {
    id: "e1",
    source: "n1",
    target: "n2",
    predicate: "KNOWS",
    weight: 1,
    validFrom: null,
    validTo: null,
    createdAt: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeEpisode(overrides: Partial<Episode> = {}): Episode {
  return {
    episodeId: "ep1",
    content: "Test episode content for verification",
    source: "mcp",
    status: "completed",
    createdAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-01T00:00:00Z",
    entities: [{ id: "n1", name: "Alice", entityType: "Person" }],
    factsCount: 2,
    processingDurationMs: 150,
    error: null,
    retryCount: 0,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("graphSlice", () => {
  it("loadInitialGraph populates nodes and edges", async () => {
    const node1 = makeNode({ id: "n1" });
    const node2 = makeNode({ id: "n2", name: "Node 2" });
    const edge = makeEdge({ id: "e1", source: "n1", target: "n2" });

    mockedApi.getNeighborhood.mockResolvedValueOnce({
      centerId: "n1",
      nodes: [node1, node2],
      edges: [edge],
      truncated: false,
      totalInNeighborhood: 2,
    });

    const store = createTestStore();
    await store.getState().loadInitialGraph();

    expect(Object.keys(store.getState().nodes)).toHaveLength(2);
    expect(Object.keys(store.getState().edges)).toHaveLength(1);
    expect(store.getState().centerNodeId).toBe("n1");
    expect(store.getState().isLoading).toBe(false);
  });

  it("expandNode merges new nodes without clearing existing", async () => {
    const store = createTestStore();

    // Pre-populate with existing data
    store.setState((s) => {
      s.nodes = { n1: makeNode({ id: "n1" }) };
      s.edges = {};
    });

    const newNode = makeNode({ id: "n3", name: "New Node" });
    const newEdge = makeEdge({ id: "e2", source: "n1", target: "n3" });

    mockedApi.getNeighbors.mockResolvedValueOnce({
      centerId: "n1",
      nodes: [newNode],
      edges: [newEdge],
      truncated: false,
      totalInNeighborhood: 1,
    });

    await store.getState().expandNode("n1");

    expect(Object.keys(store.getState().nodes)).toHaveLength(2);
    expect(store.getState().nodes["n1"]).toBeDefined();
    expect(store.getState().nodes["n3"]).toBeDefined();
    expect(Object.keys(store.getState().edges)).toHaveLength(1);
  });

  it("clear resets state", () => {
    const store = createTestStore();
    store.setState((s) => {
      s.nodes = { n1: makeNode() };
      s.edges = { e1: makeEdge() };
      s.centerNodeId = "n1";
      s.error = "some error";
    });

    store.getState().clear();

    expect(Object.keys(store.getState().nodes)).toHaveLength(0);
    expect(Object.keys(store.getState().edges)).toHaveLength(0);
    expect(store.getState().centerNodeId).toBeNull();
    expect(store.getState().error).toBeNull();
  });

  it("mergeGraphDelta adds and removes nodes/edges", () => {
    const store = createTestStore();
    store.setState((s) => {
      s.nodes = { n1: makeNode({ id: "n1" }), n2: makeNode({ id: "n2" }) };
      s.edges = { e1: makeEdge({ id: "e1" }) };
    });

    store.getState().mergeGraphDelta({
      nodesAdded: [makeNode({ id: "n3", name: "New" })],
      nodesRemoved: ["n2"],
      edgesAdded: [makeEdge({ id: "e2", source: "n1", target: "n3" })],
      edgesRemoved: ["e1"],
    });

    expect(store.getState().nodes["n1"]).toBeDefined();
    expect(store.getState().nodes["n2"]).toBeUndefined();
    expect(store.getState().nodes["n3"]).toBeDefined();
    expect(store.getState().edges["e1"]).toBeUndefined();
    expect(store.getState().edges["e2"]).toBeDefined();
  });

  it("loadGraphAt fetches graph snapshot at timestamp", async () => {
    mockedApi.getGraphAt.mockResolvedValueOnce({
      centerId: "n1",
      nodes: [makeNode({ id: "n1" })],
      edges: [],
      truncated: false,
      totalInNeighborhood: 1,
    });

    const store = createTestStore();
    await store.getState().loadGraphAt("2024-06-01T00:00:00Z");

    expect(mockedApi.getGraphAt).toHaveBeenCalledWith("2024-06-01T00:00:00Z", undefined);
    expect(Object.keys(store.getState().nodes)).toHaveLength(1);
    expect(store.getState().isLoading).toBe(false);
  });
});

describe("selectionSlice", () => {
  it("selectNode and hoverNode update state", () => {
    const store = createTestStore();

    store.getState().selectNode("n1");
    expect(store.getState().selectedNodeId).toBe("n1");

    store.getState().hoverNode("n2");
    expect(store.getState().hoveredNodeId).toBe("n2");

    store.getState().selectNode(null);
    expect(store.getState().selectedNodeId).toBeNull();
  });

  it("executeSearch populates results", async () => {
    const results: SearchResult[] = [
      { id: "n1", name: "Alice", entityType: "Person", summary: null, activationScore: 0.8 },
      { id: "n2", name: "Bob", entityType: "Person", summary: null, activationScore: 0.5 },
    ];

    mockedApi.searchEntities.mockResolvedValueOnce(results);

    const store = createTestStore();
    await store.getState().executeSearch("Alice");

    expect(store.getState().searchResults).toHaveLength(2);
    expect(store.getState().searchResults[0].name).toBe("Alice");
    expect(store.getState().isSearching).toBe(false);
  });
});

describe("preferencesSlice", () => {
  it("toggle changes state", () => {
    const store = createTestStore();

    expect(store.getState().showActivationHeatmap).toBe(true);
    store.getState().toggleActivationHeatmap();
    expect(store.getState().showActivationHeatmap).toBe(false);

    expect(store.getState().showEdgeLabels).toBe(false);
    store.getState().toggleEdgeLabels();
    expect(store.getState().showEdgeLabels).toBe(true);

    expect(store.getState().darkMode).toBe(true);
    store.getState().toggleDarkMode();
    expect(store.getState().darkMode).toBe(false);

    store.getState().setRenderMode("2d");
    expect(store.getState().renderMode).toBe("2d");
  });
});

describe("timeSlice", () => {
  it("manages time position and scrubbing state", () => {
    const store = createTestStore();

    expect(store.getState().timePosition).toBeNull();
    expect(store.getState().isTimeScrubbing).toBe(false);

    store.getState().setTimePosition("2024-06-15T12:00:00Z");
    expect(store.getState().timePosition).toBe("2024-06-15T12:00:00Z");

    store.getState().setIsTimeScrubbing(true);
    expect(store.getState().isTimeScrubbing).toBe(true);

    store.getState().setTimeRange({ start: "2024-01-01T00:00:00Z", end: "2024-12-31T23:59:59Z" });
    expect(store.getState().timeRange?.start).toBe("2024-01-01T00:00:00Z");
  });
});

describe("episodeSlice", () => {
  it("loadEpisodes populates episodes list", async () => {
    const episodes = [makeEpisode({ episodeId: "ep1" }), makeEpisode({ episodeId: "ep2" })];

    mockedApi.getEpisodes.mockResolvedValueOnce({
      items: episodes,
      nextCursor: "cursor_abc",
    });

    const store = createTestStore();
    await store.getState().loadEpisodes();

    expect(store.getState().episodes).toHaveLength(2);
    expect(store.getState().episodeCursor).toBe("cursor_abc");
    expect(store.getState().hasMoreEpisodes).toBe(true);
    expect(store.getState().isLoadingEpisodes).toBe(false);
  });

  it("loadEpisodes with cursor appends episodes", async () => {
    const store = createTestStore();
    store.setState((s) => {
      s.episodes = [makeEpisode({ episodeId: "ep1" })];
      s.episodeCursor = "cursor_1";
    });

    mockedApi.getEpisodes.mockResolvedValueOnce({
      items: [makeEpisode({ episodeId: "ep2" })],
      nextCursor: null,
    });

    await store.getState().loadEpisodes("cursor_1");

    expect(store.getState().episodes).toHaveLength(2);
    expect(store.getState().hasMoreEpisodes).toBe(false);
  });

  it("prependEpisode adds to front of list", () => {
    const store = createTestStore();
    store.setState((s) => {
      s.episodes = [makeEpisode({ episodeId: "ep1" })];
    });

    store.getState().prependEpisode(makeEpisode({ episodeId: "ep2" }));

    expect(store.getState().episodes[0].episodeId).toBe("ep2");
    expect(store.getState().episodes).toHaveLength(2);
  });

  it("updateEpisodeStatus updates matching episode", () => {
    const store = createTestStore();
    store.setState((s) => {
      s.episodes = [makeEpisode({ episodeId: "ep1", status: "processing" })];
    });

    store.getState().updateEpisodeStatus("ep1", "completed");

    expect(store.getState().episodes[0].status).toBe("completed");
  });

  it("updateEpisodeStatus with error sets error field", () => {
    const store = createTestStore();
    store.setState((s) => {
      s.episodes = [makeEpisode({ episodeId: "ep1", status: "processing" })];
    });

    store.getState().updateEpisodeStatus("ep1", "failed", "Extraction failed");

    expect(store.getState().episodes[0].status).toBe("failed");
    expect(store.getState().episodes[0].error).toBe("Extraction failed");
  });
});

describe("statsSlice", () => {
  it("loadStats populates stats", async () => {
    mockedApi.getStats.mockResolvedValueOnce({
      totalEntities: 42,
      totalRelationships: 100,
      totalEpisodes: 15,
      entityTypeCounts: { Person: 20, Organization: 10 },
      topActivated: [],
      topConnected: [{ id: "n1", name: "Alice", entityType: "Person", connectionCount: 12 }],
      growthTimeline: [{ date: "2024-01", entities: 10, episodes: 5 }],
    });

    const store = createTestStore();
    await store.getState().loadStats();

    expect(store.getState().stats?.totalEntities).toBe(42);
    expect(store.getState().stats?.topConnected).toHaveLength(1);
    expect(store.getState().isLoadingStats).toBe(false);
  });
});

describe("wsSlice", () => {
  it("manages WebSocket state", () => {
    const store = createTestStore();

    expect(store.getState().readyState).toBe("disconnected");
    expect(store.getState().lastSeq).toBe(0);

    store.getState().setReadyState("connected");
    expect(store.getState().readyState).toBe("connected");

    store.getState().setLastSeq(42);
    expect(store.getState().lastSeq).toBe(42);

    store.getState().setReconnectAttempt(3);
    expect(store.getState().reconnectAttempt).toBe(3);
  });
});

describe("knowledgeSlice — new features", () => {
  it("loadPulseEntities calls getActivationSnapshot and sets pulseEntities", async () => {
    mockedApi.getActivationSnapshot.mockResolvedValueOnce({
      topActivated: [
        { entityId: "e1", name: "Engram", entityType: "Project", currentActivation: 0.94, accessCount: 10, lastAccessedAt: null, decayRate: 0.5 },
        { entityId: "e2", name: "Konner", entityType: "Person", currentActivation: 0.91, accessCount: 8, lastAccessedAt: null, decayRate: 0.5 },
      ],
    });

    const store = createTestStore();
    await store.getState().loadPulseEntities();

    expect(mockedApi.getActivationSnapshot).toHaveBeenCalledWith(5);
    expect(store.getState().pulseEntities).toHaveLength(2);
    expect(store.getState().pulseEntities[0].name).toBe("Engram");
    expect(store.getState().pulseEntities[0].currentActivation).toBe(0.94);
    expect(store.getState().isPulseLoading).toBe(false);
  });

  it("openDrawer fetches entity detail and sets drawerEntity", async () => {
    const detail = {
      id: "e1",
      name: "Engram",
      entityType: "Project",
      summary: "Memory system",
      activationCurrent: 0.94,
      accessCount: 10,
      lastAccessed: null,
      createdAt: "2024-01-01T00:00:00Z",
      updatedAt: "2024-01-01T00:00:00Z",
      facts: [],
    };
    mockedApi.getEntity.mockResolvedValueOnce(detail);

    const store = createTestStore();
    await store.getState().openDrawer("e1");

    expect(mockedApi.getEntity).toHaveBeenCalledWith("e1");
    expect(store.getState().drawerEntityId).toBe("e1");
    expect(store.getState().drawerEntity?.name).toBe("Engram");
    expect(store.getState().isDrawerLoading).toBe(false);
  });

  it("closeDrawer clears drawer state", async () => {
    const store = createTestStore();
    store.setState((s) => {
      s.drawerEntityId = "e1";
      s.drawerEntity = {
        id: "e1",
        name: "Engram",
        entityType: "Project",
        summary: null,
        activationCurrent: 0.5,
        accessCount: 1,
        lastAccessed: null,
        createdAt: "2024-01-01T00:00:00Z",
        updatedAt: "2024-01-01T00:00:00Z",
        facts: [],
      };
    });

    store.getState().closeDrawer();

    expect(store.getState().drawerEntityId).toBeNull();
    expect(store.getState().drawerEntity).toBeNull();
  });

  it("submitInput routes 'remember X' to api.remember", async () => {
    mockedApi.remember.mockResolvedValueOnce({ status: "ok", episodeId: "ep1" });

    const store = createTestStore();
    const appendMessages = vi.fn();
    await store.getState().submitInput("remember I prefer dark mode", appendMessages);

    expect(mockedApi.remember).toHaveBeenCalledWith({
      content: "I prefer dark mode",
      source: "dashboard",
    });
    // Should call appendMessages callback with confirmation
    expect(appendMessages).toHaveBeenCalledWith(
      "remember I prefer dark mode",
      'Remembered: "I prefer dark mode"',
    );
  });

  it("submitInput routes 'forget X' to confirm dialog", async () => {
    const store = createTestStore();
    await store.getState().submitInput("forget Python");

    expect(store.getState().confirmDialog).not.toBeNull();
    expect(store.getState().confirmDialog?.type).toBe("forget");
    expect(store.getState().confirmDialog?.entityName).toBe("Python");
  });

  it("submitInput routes '/observe X' to api.observe", async () => {
    mockedApi.observe.mockResolvedValueOnce({ status: "ok", episodeId: "ep1" });

    const store = createTestStore();
    await store.getState().submitInput("/observe the weather is nice today");

    expect(mockedApi.observe).toHaveBeenCalledWith({
      content: "the weather is nice today",
      source: "dashboard",
    });
  });

  it("updateEntity calls api.updateEntity and refreshes drawer", async () => {
    const detail = {
      id: "e1",
      name: "Engram",
      entityType: "Project",
      summary: "Updated summary",
      activationCurrent: 0.94,
      accessCount: 10,
      lastAccessed: null,
      createdAt: "2024-01-01T00:00:00Z",
      updatedAt: "2024-01-01T00:00:00Z",
      facts: [],
    };
    mockedApi.updateEntity.mockResolvedValueOnce({});
    mockedApi.getEntity.mockResolvedValueOnce(detail);

    const store = createTestStore();
    store.setState((s) => { s.drawerEntityId = "e1"; });
    await store.getState().updateEntity("e1", { summary: "Updated summary" });

    expect(mockedApi.updateEntity).toHaveBeenCalledWith("e1", { summary: "Updated summary" });
    expect(mockedApi.getEntity).toHaveBeenCalledWith("e1");
    expect(store.getState().drawerEntity?.summary).toBe("Updated summary");
  });

  it("deleteEntity calls api.deleteEntity and clears drawer", async () => {
    mockedApi.deleteEntity.mockResolvedValueOnce({});
    mockedApi.getActivationSnapshot.mockResolvedValueOnce({ topActivated: [] });
    mockedApi.searchEntities.mockResolvedValueOnce([]);

    const store = createTestStore();
    store.setState((s) => {
      s.drawerEntityId = "e1";
      s.drawerEntity = {
        id: "e1",
        name: "Engram",
        entityType: "Project",
        summary: null,
        activationCurrent: 0.5,
        accessCount: 1,
        lastAccessed: null,
        createdAt: "2024-01-01T00:00:00Z",
        updatedAt: "2024-01-01T00:00:00Z",
        facts: [],
      };
    });

    await store.getState().deleteEntity("e1");

    expect(mockedApi.deleteEntity).toHaveBeenCalledWith("e1");
    expect(store.getState().drawerEntityId).toBeNull();
    expect(store.getState().drawerEntity).toBeNull();
  });

  it("confirmAction dispatches correctly for forget dialog", async () => {
    mockedApi.forget.mockResolvedValueOnce({ status: "ok" });
    mockedApi.getActivationSnapshot.mockResolvedValueOnce({ topActivated: [] });
    mockedApi.searchEntities.mockResolvedValueOnce([]);

    const store = createTestStore();
    store.setState((s) => {
      s.confirmDialog = {
        type: "forget",
        entityName: "Python",
        title: "Forget Entity",
        message: "Are you sure?",
      };
    });

    await store.getState().confirmAction();

    expect(mockedApi.forget).toHaveBeenCalledWith({ entity_name: "Python" });
    expect(store.getState().confirmDialog).toBeNull();
  });

  it("setPulseEntities updates pulse state directly", () => {
    const store = createTestStore();
    store.getState().setPulseEntities([
      { entityId: "e1", name: "Test", entityType: "Concept", currentActivation: 0.5 },
    ]);
    expect(store.getState().pulseEntities).toHaveLength(1);
    expect(store.getState().pulseEntities[0].name).toBe("Test");
  });

  it("setSearchOverlayOpen and setBrowseOverlayOpen toggle overlays", () => {
    const store = createTestStore();

    expect(store.getState().searchOverlayOpen).toBe(false);
    store.getState().setSearchOverlayOpen(true);
    expect(store.getState().searchOverlayOpen).toBe(true);

    expect(store.getState().browseOverlayOpen).toBe(false);
    store.getState().setBrowseOverlayOpen(true);
    expect(store.getState().browseOverlayOpen).toBe(true);
  });
});
