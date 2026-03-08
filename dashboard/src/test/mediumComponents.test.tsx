import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mock force graph components
vi.mock("react-force-graph-3d", () => ({
  default: (props: Record<string, unknown>) => (
    <div data-testid="force-graph-3d" data-background-color={props.backgroundColor as string} />
  ),
}));

vi.mock("react-force-graph-2d", () => ({
  default: (props: Record<string, unknown>) => (
    <div data-testid="force-graph-2d" data-background-color={props.backgroundColor as string} />
  ),
}));

// Mock recharts to avoid canvas/SVG rendering issues in jsdom
vi.mock("recharts", () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="xaxis" />,
  YAxis: () => <div data-testid="yaxis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  ReferenceLine: () => <div data-testid="reference-line" />,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
}));

vi.mock("../api/client", () => ({
  api: {
    getGraphAtlas: vi.fn().mockResolvedValue({
      representation: {
        scope: "atlas",
        layout: "precomputed",
        representedEntityCount: 1,
        representedEdgeCount: 0,
        displayedNodeCount: 1,
        displayedEdgeCount: 0,
        truncated: false,
      },
      generatedAt: "2026-03-06T00:00:00Z",
      regions: [
        {
          id: "region:test",
          label: "People",
          subtitle: "Dominant entity type: Person",
          kind: "mixed",
          memberCount: 1,
          representedEdgeCount: 0,
          activationScore: 0.5,
          growth7d: 1,
          growth30d: 1,
          dominantEntityTypes: { Person: 1 },
          hubEntityIds: ["n1"],
          centerEntityId: "n1",
          latestEntityCreatedAt: "2026-03-05T00:00:00Z",
          x: 0,
          y: 0,
          z: 0,
        },
      ],
      bridges: [],
      stats: {
        totalEntities: 1,
        totalRelationships: 0,
        totalRegions: 1,
        hottestRegionId: "region:test",
        fastestGrowingRegionId: "region:test",
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
    getHealth: vi.fn().mockResolvedValue({
      status: "healthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "healthy" },
    }),
    getNeighborhood: vi.fn().mockResolvedValue({
      centerId: "n1",
      nodes: [],
      edges: [],
      truncated: false,
      totalInNeighborhood: 0,
    }),
    getNeighbors: vi.fn(),
    searchEntities: vi.fn().mockResolvedValue([]),
    getEntity: vi.fn(),
    getStats: vi.fn().mockResolvedValue({
      totalEntities: 10,
      totalRelationships: 20,
      totalEpisodes: 5,
      entityTypeCounts: { Person: 5, Organization: 3 },
      topActivated: [],
      topConnected: [],
      growthTimeline: [],
    }),
    getEpisodes: vi.fn().mockResolvedValue({ items: [], nextCursor: null }),
    getGraphAt: vi.fn(),
    updateEntity: vi.fn(),
    deleteEntity: vi.fn(),
    getActivationSnapshot: vi.fn().mockResolvedValue({ topActivated: [] }),
    getActivationCurve: vi.fn(),
  },
}));

// Mock WebSocket
vi.stubGlobal("WebSocket", vi.fn(() => ({
  close: vi.fn(),
  send: vi.fn(),
  readyState: 1,
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
})));

// Mock sendWsCommand to avoid WebSocket side effects
vi.mock("../hooks/useWebSocket", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../hooks/useWebSocket")>();
  return {
    ...actual,
    useWebSocket: vi.fn(),
    sendWsCommand: vi.fn().mockReturnValue(true),
    getWsInstance: vi.fn().mockReturnValue(null),
  };
});

import { ActivationMonitor } from "../components/ActivationMonitor";
import { Sidebar } from "../components/Sidebar";
import { DashboardShell } from "../components/DashboardShell";
import { EmptyState } from "../components/EmptyState";
import { SearchBar } from "../components/SearchBar";
import { GraphControls } from "../components/GraphControls";
import { NodeTooltip } from "../components/NodeTooltip";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { api } from "../api/client";
import { useEngramStore } from "../store";

function resetStore() {
  useEngramStore.setState({
    nodes: {},
    edges: {},
    centerNodeId: null,
    brainMapScope: "atlas",
    representation: null,
    atlas: null,
    activeRegionId: null,
    regionData: null,
    lastAtlasVisitAt: null,
    lastAtlasSnapshotId: null,
    isLoading: false,
    error: null,
    selectedNodeId: null,
    hoveredNodeId: null,
    selectedEdgeId: null,
    searchQuery: "",
    searchResults: [],
    isSearching: false,
    currentView: "graph",
    renderMode: "3d",
    showActivationHeatmap: true,
    showEdgeLabels: false,
    darkMode: true,
    graphMaxNodes: 2000,
    timePosition: null,
    timeRange: null,
    isTimeScrubbing: false,
    atlasSnapshotId: null,
    atlasHistory: [],
    episodes: [],
    episodeCursor: null,
    hasMoreEpisodes: true,
    isLoadingEpisodes: false,
    stats: null,
    isLoadingStats: false,
    readyState: "disconnected",
    lastSeq: 0,
    reconnectAttempt: 0,
    activationLeaderboard: [],
    selectedActivationEntity: null,
    decayCurve: [],
    decayFormula: "",
    accessEvents: [],
    isActivationSubscribed: false,
    isLoadingCurve: false,
  });
}

beforeEach(() => {
  act(() => {
    resetStore();
  });
});

// --- ActivationMonitor ---

describe("ActivationMonitor", () => {
  it("renders leaderboard with entities", () => {
    act(() => {
      useEngramStore.setState({
        activationLeaderboard: [
          { entityId: "e1", name: "Alice", entityType: "Person", currentActivation: 0.8, accessCount: 5, lastAccessedAt: null, decayRate: 0.5 },
          { entityId: "e2", name: "Bob", entityType: "Person", currentActivation: 0.5, accessCount: 3, lastAccessedAt: null, decayRate: 0.5 },
        ],
      });
    });
    render(<ActivationMonitor />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("0.80")).toBeInTheDocument();
  });

  it("renders empty state when no entities", () => {
    render(<ActivationMonitor />);
    expect(screen.getByText("No activated entities")).toBeInTheDocument();
  });

  it("shows decay curve when entity is selected", () => {
    act(() => {
      useEngramStore.setState({
        activationLeaderboard: [
          { entityId: "e1", name: "Alice", entityType: "Person", currentActivation: 0.8, accessCount: 5, lastAccessedAt: null, decayRate: 0.5 },
        ],
        selectedActivationEntity: "e1",
        decayCurve: [
          { timestamp: new Date().toISOString(), activation: 0.8 },
        ],
        decayFormula: "B_i = ln(Σ t_j^{-0.5})",
      });
    });
    render(<ActivationMonitor />);
    expect(screen.getByText("Activation Decay Curve")).toBeInTheDocument();
    expect(screen.getByText(/B_i/)).toBeInTheDocument();
  });

  it("toggles LIVE/PAUSED subscription", async () => {
    const user = userEvent.setup();
    act(() => {
      useEngramStore.setState({ readyState: "connected" });
    });
    render(<ActivationMonitor />);

    const button = screen.getByText("PAUSED");
    expect(button).toBeInTheDocument();

    await user.click(button);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(useEngramStore.getState().isActivationSubscribed).toBe(true);
  });
});

// --- Sidebar ---

describe("Sidebar", () => {
  it("renders 6 nav items", () => {
    render(<Sidebar />);
    expect(screen.getByText("Graph")).toBeInTheDocument();
    expect(screen.getByText("Timeline")).toBeInTheDocument();
    expect(screen.getByText("Feed")).toBeInTheDocument();
    expect(screen.getByText("Activation")).toBeInTheDocument();
    expect(screen.getByText("Stats")).toBeInTheDocument();
    expect(screen.getByText("Consolidate")).toBeInTheDocument();
  });

  it("highlights active view", () => {
    act(() => {
      useEngramStore.setState({ currentView: "feed" });
    });
    render(<Sidebar />);
    const feedButton = screen.getByText("Feed").closest("button");
    expect(feedButton?.style.background).toContain("rgba(34, 211, 238, 0.08)");
  });

  it("clicking view calls setCurrentView", async () => {
    const user = userEvent.setup();
    render(<Sidebar />);

    await user.click(screen.getByText("Stats"));
    expect(useEngramStore.getState().currentView).toBe("stats");
  });
});

// --- DashboardShell ---

describe("DashboardShell", () => {
  it("renders sidebar and atlas metadata", async () => {
    await act(async () => {
      await useEngramStore.getState().loadAtlas();
    });
    render(<DashboardShell />);
    expect(screen.getByText("Engram")).toBeInTheDocument();
    expect(screen.getByText("Atlas")).toBeInTheDocument();
    expect(screen.getByText(/regions/)).toBeInTheDocument();
  });

  it("switches view on currentView change", async () => {
    act(() => {
      useEngramStore.setState({ currentView: "feed" });
    });
    render(<DashboardShell />);
    expect(await screen.findByText("Loading view...")).toBeInTheDocument();
    expect(await screen.findByText("Filter")).toBeInTheDocument();
  });
});

// --- EmptyState ---

describe("EmptyState (additional)", () => {
  it("renders title and description", () => {
    render(<EmptyState />);
    expect(screen.getByText("No memories yet")).toBeInTheDocument();
    expect(screen.getByText(/Use the MCP tools/)).toBeInTheDocument();
  });

  it("renders awaiting input badge", () => {
    render(<EmptyState />);
    expect(screen.getByText("awaiting input")).toBeInTheDocument();
  });
});

// --- SearchBar ---

describe("SearchBar (additional)", () => {
  it("renders input element", () => {
    render(<SearchBar />);
    expect(screen.getByPlaceholderText("Search entities...")).toBeInTheDocument();
  });

  it("debounces search calls", async () => {
    const user = userEvent.setup();
    render(<SearchBar />);

    const input = screen.getByPlaceholderText("Search entities...");
    await user.type(input, "test");

    expect(useEngramStore.getState().searchQuery).toBe("test");
  });
});

// --- GraphControls ---

describe("GraphControls", () => {
  it("renders toggle buttons", () => {
    render(<GraphControls />);
    expect(screen.getByText("3D")).toBeInTheDocument();
    expect(screen.getByText("Heatmap")).toBeInTheDocument();
    expect(screen.getByText("Labels")).toBeInTheDocument();
  });

  it("clicking 2D/3D toggles render mode", async () => {
    const user = userEvent.setup();
    render(<GraphControls />);

    expect(useEngramStore.getState().renderMode).toBe("3d");
    await user.click(screen.getByText("3D"));
    expect(useEngramStore.getState().renderMode).toBe("2d");
  });
});

// --- NodeTooltip ---

describe("NodeTooltip (additional)", () => {
  it("renders entity name and type", () => {
    act(() => {
      useEngramStore.setState({
        hoveredNodeId: "n1",
        nodes: {
          n1: {
            id: "n1",
            name: "Alice",
            entityType: "Person",
            summary: null,
            activationCurrent: 0.75,
            accessCount: 5,
            lastAccessed: "2024-01-01T00:00:00Z",
            createdAt: "2024-01-01T00:00:00Z",
            updatedAt: "2024-01-01T00:00:00Z",
          },
        },
      });
    });
    render(<NodeTooltip />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Person")).toBeInTheDocument();
  });

  it("renders activation percentage", () => {
    act(() => {
      useEngramStore.setState({
        hoveredNodeId: "n1",
        nodes: {
          n1: {
            id: "n1",
            name: "Bob",
            entityType: "Organization",
            summary: null,
            activationCurrent: 0.5,
            accessCount: 2,
            lastAccessed: null,
            createdAt: "2024-01-01T00:00:00Z",
            updatedAt: "2024-01-01T00:00:00Z",
          },
        },
      });
    });
    render(<NodeTooltip />);
    expect(screen.getByText("50%")).toBeInTheDocument();
  });
});

// --- ConnectionStatus ---

describe("ConnectionStatus (additional)", () => {
  it("shows Live when ws open", async () => {
    act(() => {
      useEngramStore.setState({ readyState: "connected" });
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Live")).toBeInTheDocument();
  });

  it("shows Offline when health checks fail", async () => {
    vi.mocked(api.getHealth).mockResolvedValueOnce({
      status: "unhealthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "unhealthy" },
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Offline")).toBeInTheDocument();
  });
});
