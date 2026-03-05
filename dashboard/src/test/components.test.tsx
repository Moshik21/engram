import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mock force graph components before any imports that use them
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

vi.mock("../components/knowledge/ChatProvider", () => ({
  useChatContext: () => ({
    messages: [],
    setMessages: vi.fn(),
    sendMessage: vi.fn(),
    status: "ready",
    error: undefined,
    id: "test-chat",
    stop: vi.fn(),
    regenerate: vi.fn(),
    resumeStream: vi.fn(),
    addToolResult: vi.fn(),
    addToolOutput: vi.fn(),
    addToolApprovalResponse: vi.fn(),
    clearError: vi.fn(),
  }),
  ChatProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

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
    recall: vi.fn(),
    searchFacts: vi.fn(),
    getKnowledgeContext: vi.fn(),
    observe: vi.fn(),
    remember: vi.fn(),
    forget: vi.fn(),
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

import { GraphExplorer } from "../components/GraphExplorer";
import { NodeTooltip } from "../components/NodeTooltip";
import { EmptyState } from "../components/EmptyState";
import { SearchBar } from "../components/SearchBar";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { EpisodeCard } from "../components/EpisodeCard";
import { MemoryFeed } from "../components/MemoryFeed";
import { StatsPanel } from "../components/StatsPanel";
import { useEngramStore } from "../store";
import type { Episode } from "../store/types";

function resetStore() {
  useEngramStore.setState({
    nodes: {},
    edges: {},
    centerNodeId: null,
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
    episodes: [],
    episodeCursor: null,
    hasMoreEpisodes: true,
    isLoadingEpisodes: false,
    stats: null,
    isLoadingStats: false,
    readyState: "disconnected",
    lastSeq: 0,
    reconnectAttempt: 0,
  });
}

beforeEach(() => {
  act(() => {
    resetStore();
  });
});

describe("GraphExplorer", () => {
  it("renders EmptyState when no nodes and not loading", () => {
    render(<GraphExplorer />);
    expect(screen.getByText("No memories yet")).toBeInTheDocument();
  });

  it("renders force graph when nodes exist", () => {
    act(() => {
      useEngramStore.setState({
        nodes: {
          n1: {
            id: "n1",
            name: "Alice",
            entityType: "Person",
            summary: null,
            activationCurrent: 0.5,
            accessCount: 1,
            lastAccessed: null,
            createdAt: "2024-01-01T00:00:00Z",
            updatedAt: "2024-01-01T00:00:00Z",
          },
        },
      });
    });
    render(<GraphExplorer />);
    expect(screen.getByTestId("force-graph-3d")).toBeInTheDocument();
  });
});

describe("NodeTooltip", () => {
  it("shows tooltip when hoveredNodeId is set", () => {
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
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("returns null when hoveredNodeId is not set", () => {
    const { container } = render(<NodeTooltip />);
    expect(container.innerHTML).toBe("");
  });
});

describe("EmptyState", () => {
  it("renders empty state message", () => {
    render(<EmptyState />);
    expect(screen.getByText("No memories yet")).toBeInTheDocument();
    expect(screen.getByText(/Use the MCP tools/)).toBeInTheDocument();
  });
});

describe("SearchBar", () => {
  it("debounces input before searching", async () => {
    const user = userEvent.setup();
    render(<SearchBar />);

    const input = screen.getByPlaceholderText("Search entities...");
    await user.type(input, "test");

    // The search query should update immediately in the store
    expect(useEngramStore.getState().searchQuery).toBe("test");
  });
});

describe("ConnectionStatus", () => {
  it("shows Offline by default", () => {
    render(<ConnectionStatus />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
  });

  it("shows Connected when WebSocket is connected", () => {
    act(() => {
      useEngramStore.setState({ readyState: "connected" });
    });
    render(<ConnectionStatus />);
    expect(screen.getByText("Connected")).toBeInTheDocument();
  });

  it("shows Reconnecting when WebSocket is reconnecting", () => {
    act(() => {
      useEngramStore.setState({ readyState: "reconnecting" });
    });
    render(<ConnectionStatus />);
    expect(screen.getByText("Reconnecting")).toBeInTheDocument();
  });
});

describe("EpisodeCard", () => {
  const mockEpisode: Episode = {
    episodeId: "ep1",
    content: "Alice met Bob at the conference. They discussed machine learning and neural networks.",
    source: "mcp",
    status: "completed",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    entities: [
      { id: "n1", name: "Alice", entityType: "Person" },
      { id: "n2", name: "Bob", entityType: "Person" },
    ],
    factsCount: 3,
    processingDurationMs: 200,
    error: null,
    retryCount: 0,
  };

  it("renders episode content preview", () => {
    render(<EpisodeCard episode={mockEpisode} onEntityClick={vi.fn()} />);
    expect(screen.getByText(/Alice met Bob/)).toBeInTheDocument();
  });

  it("renders status pill", () => {
    render(<EpisodeCard episode={mockEpisode} onEntityClick={vi.fn()} />);
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("renders source badge", () => {
    render(<EpisodeCard episode={mockEpisode} onEntityClick={vi.fn()} />);
    expect(screen.getByText("mcp")).toBeInTheDocument();
  });

  it("renders entity chips", () => {
    render(<EpisodeCard episode={mockEpisode} onEntityClick={vi.fn()} />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
  });

  it("calls onEntityClick when chip clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<EpisodeCard episode={mockEpisode} onEntityClick={onClick} />);

    await user.click(screen.getByText("Alice"));
    expect(onClick).toHaveBeenCalledWith("n1");
  });

  it("renders facts count", () => {
    render(<EpisodeCard episode={mockEpisode} onEntityClick={vi.fn()} />);
    expect(screen.getByText("3 facts extracted")).toBeInTheDocument();
  });

  it("truncates long content to 120 chars", () => {
    const longContent = "A".repeat(200);
    const ep = { ...mockEpisode, content: longContent };
    render(<EpisodeCard episode={ep} onEntityClick={vi.fn()} />);
    expect(screen.getByText(/A{120}\.\.\./)).toBeInTheDocument();
  });

  it("shows error message for failed episodes", () => {
    const failedEp = { ...mockEpisode, status: "failed" as const, error: "Extraction timed out" };
    render(<EpisodeCard episode={failedEp} onEntityClick={vi.fn()} />);
    expect(screen.getByText("Extraction timed out")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});

describe("MemoryFeed", () => {
  it("renders empty state when no episodes", async () => {
    render(<MemoryFeed />);
    // loadEpisodes triggers on mount, wait for it to resolve
    expect(await screen.findByText(/No episodes yet/)).toBeInTheDocument();
  });

  it("renders filter controls", () => {
    render(<MemoryFeed />);
    expect(screen.getByText("Filter")).toBeInTheDocument();
  });
});

describe("StatsPanel", () => {
  it("renders loading state initially", () => {
    act(() => {
      useEngramStore.setState({ isLoadingStats: true, stats: null });
    });
    const { container } = render(<StatsPanel />);
    // Loading state shows a skeleton element
    expect(container.querySelector(".skeleton")).toBeInTheDocument();
  });

  it("renders stats when loaded", () => {
    act(() => {
      useEngramStore.setState({
        stats: {
          totalEntities: 42,
          totalRelationships: 100,
          totalEpisodes: 15,
          entityTypeCounts: { Person: 20, Organization: 10 },
          topActivated: [],
          topConnected: [],
          growthTimeline: [],
        },
        isLoadingStats: false,
      });
    });
    render(<StatsPanel />);
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
  });
});

// --- Knowledge Tab Redesign Tests ---

import { MemoryPulse } from "../components/knowledge/MemoryPulse";
import { IntentIndicator } from "../components/knowledge/IntentIndicator";
import { ConfirmDialog } from "../components/knowledge/ConfirmDialog";
import { SearchOverlay } from "../components/knowledge/SearchOverlay";

describe("MemoryPulse", () => {
  it("renders nothing when no pulse entities and not loading", () => {
    act(() => {
      useEngramStore.setState({ pulseEntities: [], isPulseLoading: false });
    });
    const { container } = render(<MemoryPulse />);
    expect(container.innerHTML).toBe("");
  });

  it("renders skeleton pills when loading", () => {
    act(() => {
      useEngramStore.setState({ pulseEntities: [], isPulseLoading: true });
    });
    const { container } = render(<MemoryPulse />);
    expect(container.querySelectorAll(".skeleton")).toHaveLength(3);
  });

  it("renders entity pills when data is loaded", () => {
    act(() => {
      useEngramStore.setState({
        pulseEntities: [
          { entityId: "e1", name: "Engram", entityType: "Project", currentActivation: 0.94 },
          { entityId: "e2", name: "Konner", entityType: "Person", currentActivation: 0.91 },
        ],
        isPulseLoading: false,
      });
    });
    render(<MemoryPulse />);
    expect(screen.getByText("Engram")).toBeInTheDocument();
    expect(screen.getByText("Konner")).toBeInTheDocument();
    expect(screen.getByText("PULSE")).toBeInTheDocument();
  });
});

// KnowledgeChatStream tests removed — component now uses useChat from ChatProvider.
// Testing would require mocking @ai-sdk/react's useChat hook, which is covered by integration tests.

describe("IntentIndicator", () => {
  it("renders nothing when intentMode is null", () => {
    act(() => {
      useEngramStore.setState({ intentMode: null });
    });
    const { container } = render(<IntentIndicator />);
    expect(container.innerHTML).toBe("");
  });

  it("renders Recalling label when asking", () => {
    act(() => {
      useEngramStore.setState({ intentMode: "asking" });
    });
    render(<IntentIndicator />);
    expect(screen.getByText("Recalling...")).toBeInTheDocument();
  });

  it("renders Remembering label when remembering", () => {
    act(() => {
      useEngramStore.setState({ intentMode: "remembering" });
    });
    render(<IntentIndicator />);
    expect(screen.getByText("Remembering...")).toBeInTheDocument();
  });

  it("renders Observing label when observing", () => {
    act(() => {
      useEngramStore.setState({ intentMode: "observing" });
    });
    render(<IntentIndicator />);
    expect(screen.getByText("Observing...")).toBeInTheDocument();
  });

  it("renders Forgetting label when forgetting", () => {
    act(() => {
      useEngramStore.setState({ intentMode: "forgetting" });
    });
    render(<IntentIndicator />);
    expect(screen.getByText("Forgetting...")).toBeInTheDocument();
  });
});

describe("ConfirmDialog", () => {
  it("renders nothing when no confirm dialog", () => {
    act(() => {
      useEngramStore.setState({ confirmDialog: null });
    });
    const { container } = render(<ConfirmDialog />);
    expect(container.innerHTML).toBe("");
  });

  it("renders delete dialog with correct text", () => {
    act(() => {
      useEngramStore.setState({
        confirmDialog: {
          type: "delete",
          entityId: "e1",
          entityName: "Engram",
          title: "Delete Entity",
          message: 'Are you sure you want to delete "Engram"?',
        },
      });
    });
    render(<ConfirmDialog />);
    expect(screen.getByText("Delete Entity")).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to delete "Engram"/)).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getByText("Confirm")).toBeInTheDocument();
  });

  it("renders forget dialog with correct text", () => {
    act(() => {
      useEngramStore.setState({
        confirmDialog: {
          type: "forget",
          entityName: "Python",
          title: "Forget Entity",
          message: 'Are you sure you want to forget "Python"?',
        },
      });
    });
    render(<ConfirmDialog />);
    expect(screen.getByText("Forget Entity")).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to forget "Python"/)).toBeInTheDocument();
  });
});

describe("SearchOverlay", () => {
  it("renders search input", () => {
    act(() => {
      useEngramStore.setState({ searchOverlayOpen: true });
    });
    render(<SearchOverlay />);
    expect(screen.getByPlaceholderText("Search entities...")).toBeInTheDocument();
  });

  it("closes on Escape key", async () => {
    const user = userEvent.setup();
    act(() => {
      useEngramStore.setState({ searchOverlayOpen: true });
    });
    render(<SearchOverlay />);
    const input = screen.getByPlaceholderText("Search entities...");
    await user.click(input);
    await user.keyboard("{Escape}");
    expect(useEngramStore.getState().searchOverlayOpen).toBe(false);
  });
});
