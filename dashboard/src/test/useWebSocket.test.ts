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
});

// Must import after mocks are set up
import { useWebSocket } from "../hooks/useWebSocket";
import { useEngramStore } from "../store";

describe("useWebSocket", () => {
  it("connects on mount and disconnects on unmount", () => {
    const { unmount } = renderHook(() => useWebSocket());

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

  it("reconnects with exponential backoff on close", () => {
    renderHook(() => useWebSocket());

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
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    // Should have created a second WebSocket for reconnect
    expect(wsInstances.length).toBeGreaterThanOrEqual(2);
  });

  it("routes incoming messages to correct store actions", () => {
    renderHook(() => useWebSocket());

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
});
