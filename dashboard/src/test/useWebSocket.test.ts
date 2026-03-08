import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

// Track created WebSocket instances
interface MockWs {
  url: string;
  close: ReturnType<typeof vi.fn>;
  send: ReturnType<typeof vi.fn>;
  onopen: ((ev: Event) => void) | null;
  onclose: ((ev: CloseEvent) => void) | null;
  onmessage: ((ev: MessageEvent) => void) | null;
  onerror: ((ev: Event) => void) | null;
  readyState: number;
  OPEN: number;
  CLOSED: number;
}

let wsInstances: MockWs[];

vi.mock("../api/client", () => ({
  api: {
    getHealth: vi.fn(),
    getGraphAtlas: vi.fn().mockResolvedValue({
      representation: {
        scope: "atlas",
        layout: "precomputed",
        representedEntityCount: 0,
        representedEdgeCount: 0,
        displayedNodeCount: 0,
        displayedEdgeCount: 0,
        truncated: false,
      },
      generatedAt: "2026-03-06T00:00:00Z",
      regions: [],
      bridges: [],
      stats: {
        totalEntities: 0,
        totalRelationships: 0,
        totalRegions: 0,
        hottestRegionId: null,
        fastestGrowingRegionId: null,
      },
    }),
    getGraphAtlasHistory: vi.fn().mockResolvedValue({ items: [] }),
    getGraphRegion: vi.fn().mockResolvedValue({
      representation: {
        scope: "region",
        layout: "precomputed",
        representedEntityCount: 0,
        representedEdgeCount: 0,
        displayedNodeCount: 0,
        displayedEdgeCount: 0,
        truncated: false,
      },
      generatedAt: "2026-03-06T00:00:00Z",
      region: {
        id: "region:test",
        label: "People",
        subtitle: null,
        kind: "mixed",
        memberCount: 0,
        activationScore: 0,
        growth7d: 0,
        growth30d: 0,
        latestEntityCreatedAt: null,
      },
      nodes: [],
      edges: [],
      topEntities: [],
      memberIds: [],
    }),
    getNeighborhood: vi.fn().mockResolvedValue({
      centerId: "n1",
      nodes: [],
      edges: [],
      truncated: false,
      totalInNeighborhood: 0,
    }),
    getNeighbors: vi.fn(),
    searchEntities: vi.fn(),
    getEntity: vi.fn(),
    getStats: vi.fn(),
    getEpisodes: vi.fn().mockResolvedValue({ items: [], nextCursor: null }),
    getGraphAt: vi.fn(),
    updateEntity: vi.fn(),
    deleteEntity: vi.fn(),
    getActivationSnapshot: vi.fn(),
    getActivationCurve: vi.fn(),
  },
  getAuthToken: vi.fn().mockResolvedValue(null),
  setAuthTokenGetter: vi.fn(),
}));

beforeEach(() => {
  wsInstances = [];
  vi.useFakeTimers();

  class MockWebSocket {
    static OPEN = 1;
    static CLOSED = 3;

    url: string;
    close = vi.fn();
    send = vi.fn();
    onopen: ((ev: Event) => void) | null = null;
    onclose: ((ev: CloseEvent) => void) | null = null;
    onmessage: ((ev: MessageEvent) => void) | null = null;
    onerror: ((ev: Event) => void) | null = null;
    readyState = 0;
    OPEN = 1;
    CLOSED = 3;

    constructor(url: string) {
      this.url = url;
      wsInstances.push(this);
    }
  }

  vi.stubGlobal("WebSocket", MockWebSocket);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

// Must import after mocks are set up
import { useWebSocket, _flushActivationBatch } from "../hooks/useWebSocket";
import { useEngramStore } from "../store";

describe("useWebSocket", () => {
  it("connects on mount and disconnects on unmount", async () => {
    const { unmount } = renderHook(() => useWebSocket());

    // Flush the async connect() (awaits getAuthToken)
    await act(async () => {});

    // Should have created a WebSocket
    expect(wsInstances).toHaveLength(1);
    expect(wsInstances[0].url).toContain("/ws/dashboard");

    // Simulate open
    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    expect(useEngramStore.getState().readyState).toBe("connected");

    // Unmount should close
    unmount();
    expect(wsInstances[0].close).toHaveBeenCalled();
    expect(useEngramStore.getState().readyState).toBe("disconnected");
  });

  it("normalizes VITE_WS_URL values that already include /ws", async () => {
    vi.stubEnv("VITE_WS_URL", "ws://localhost:8100/ws");

    renderHook(() => useWebSocket());

    await act(async () => {});

    expect(wsInstances).toHaveLength(1);
    expect(wsInstances[0].url).toBe("ws://localhost:8100/ws/dashboard");
  });

  it("reconnects with exponential backoff on close", async () => {
    renderHook(() => useWebSocket());

    // Flush the async connect()
    await act(async () => {});

    // Simulate open then close
    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    act(() => {
      wsInstances[0].onclose?.(new CloseEvent("close"));
    });

    expect(useEngramStore.getState().readyState).toBe("disconnected");

    // Advance past the first backoff delay (BASE_DELAY=1000ms + jitter)
    // The reconnect calls async connect() again, so flush microtasks too
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });

    // Should have created a second WebSocket for reconnect
    expect(wsInstances.length).toBeGreaterThanOrEqual(2);
  });

  it("routes incoming messages to correct store actions", async () => {
    renderHook(() => useWebSocket());

    // Flush the async connect()
    await act(async () => {});

    // Simulate open
    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    // Send an episode.queued event
    act(() => {
      wsInstances[0].onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "episode.queued",
            seq: 1,
            episode: {
              episodeId: "ep1",
              content: "Test content",
              source: "mcp",
              status: "queued",
              createdAt: "2024-01-01T00:00:00Z",
              updatedAt: "2024-01-01T00:00:00Z",
              entities: [],
              factsCount: 0,
              processingDurationMs: null,
              error: null,
              retryCount: 0,
            },
          }),
        }),
      );
    });

    expect(useEngramStore.getState().lastSeq).toBe(1);
    expect(useEngramStore.getState().episodes).toHaveLength(1);
    expect(useEngramStore.getState().episodes[0].episodeId).toBe("ep1");
  });

  it("routes activation.access to addActivationPulse", async () => {
    renderHook(() => useWebSocket());

    // Flush the async connect()
    await act(async () => {});

    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    act(() => {
      wsInstances[0].onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "activation.access",
            seq: 2,
            // Server WebSocket handler flattens payload into top-level keys
            entityId: "ent_abc",
            name: "Alice",
            entityType: "Person",
            activation: 0.85,
            accessedVia: "recall",
          }),
        }),
      );
    });

    expect(useEngramStore.getState().lastSeq).toBe(2);

    // Activation updates are batched via requestAnimationFrame — flush synchronously
    act(() => {
      _flushActivationBatch();
    });

    const pulses = useEngramStore.getState().activationPulses;
    expect(pulses).toHaveLength(1);
    expect(pulses[0].entityId).toBe("ent_abc");
    expect(pulses[0].name).toBe("Alice");
    expect(pulses[0].accessedVia).toBe("recall");
  });

  it("routes graph.nodes_added with full data to mergeGraphDelta in neighborhood scope", async () => {
    renderHook(() => useWebSocket());

    // Flush the async connect()
    await act(async () => {});

    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    act(() => {
      useEngramStore.setState({
        brainMapScope: "neighborhood",
        representation: {
          scope: "neighborhood",
          layout: "force",
          representedEntityCount: 0,
          representedEdgeCount: 0,
          displayedNodeCount: 0,
          displayedEdgeCount: 0,
          truncated: false,
        },
      });
    });

    act(() => {
      wsInstances[0].onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "graph.nodes_added",
            seq: 3,
            // Server WebSocket handler flattens payload into top-level keys
            entity_count: 1,
            new_entities: ["React"],
            nodes: [
              {
                id: "ent_react",
                name: "React",
                entityType: "Technology",
                summary: "UI library",
                activationCurrent: 0.5,
                accessCount: 1,
                lastAccessed: null,
                createdAt: "2024-01-01T00:00:00Z",
                updatedAt: "2024-01-01T00:00:00Z",
              },
            ],
            edges: [],
          }),
        }),
      );
    });

    expect(useEngramStore.getState().lastSeq).toBe(3);
    // Node should have been merged into graph store
    expect(useEngramStore.getState().nodes["ent_react"]).toBeDefined();
    expect(useEngramStore.getState().nodes["ent_react"].name).toBe("React");
  });

  it("routes activation.snapshot to updateNodeActivations", async () => {
    renderHook(() => useWebSocket());

    // Pre-populate a node in the graph
    useEngramStore.getState().mergeGraphDelta({
      nodesAdded: [
        {
          id: "ent_x",
          name: "X",
          entityType: "Other",
          summary: null,
          activationCurrent: 0.1,
          accessCount: 0,
          lastAccessed: null,
          createdAt: "2024-01-01T00:00:00Z",
          updatedAt: "2024-01-01T00:00:00Z",
        },
      ],
    });

    // Flush the async connect()
    await act(async () => {});

    act(() => {
      wsInstances[0].readyState = 1;
      wsInstances[0].onopen?.(new Event("open"));
    });

    act(() => {
      wsInstances[0].onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "activation.snapshot",
            seq: 4,
            payload: {
              topActivated: [
                {
                  entityId: "ent_x",
                  name: "X",
                  entityType: "Other",
                  currentActivation: 0.9,
                  accessCount: 5,
                },
              ],
            },
          }),
        }),
      );
    });

    // Node activation should be synced
    expect(useEngramStore.getState().nodes["ent_x"].activationCurrent).toBe(0.9);
  });
});
