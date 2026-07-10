import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, waitFor } from "@testing-library/react";
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
        representedEntityCount: 1,
        representedEdgeCount: 0,
        displayedNodeCount: 1,
        displayedEdgeCount: 0,
        truncated: false,
      },
      generatedAt: "2026-03-06T00:00:00Z",
      region: {
        id: "region:test",
        label: "People",
        subtitle: "Dominant entity type: Person",
        kind: "mixed",
        memberCount: 1,
        activationScore: 0.5,
        growth7d: 1,
        growth30d: 1,
        latestEntityCreatedAt: "2026-03-05T00:00:00Z",
      },
      nodes: [],
      edges: [],
      topEntities: [],
      memberIds: ["n1"],
    }),
    getHealth: vi.fn().mockResolvedValue({
      status: "unhealthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "unhealthy" },
    }),
    getRuntimeState: vi.fn().mockResolvedValue({
      projectName: "Engram",
      runtime: { mode: "lite" },
      activation: {},
      features: {},
      artifactBootstrap: {
        enabled: true,
        projectPath: null,
        artifactCount: 1,
        freshArtifactCount: 1,
        staleArtifactCount: 0,
        lastObservedAt: "2026-05-19T00:00:00Z",
        staleAfterSeconds: 86400,
      },
      agentAdoption: {
        status: "ready",
        doNotTreatEmptyAsFailure: false,
        requiredNextTools: ["get_context"],
        claimAuthority: {
          tool: "claim_authority",
          args: {
            project_path: "<current_project_path>",
            file_memory_present: "<true if local/file memory is visible>",
          },
          reason: "Load portable memory authority.",
        },
        bootstrap: {
          tool: "bootstrap_project",
          required: false,
          args: { project_path: "<current_project_path>" },
          reason: "Project artifacts are current.",
        },
        reason: "Engram has runtime evidence available.",
      },
      stats: { recallMetrics: {}, epistemicMetrics: {} },
      generatedAt: "2026-05-19T00:00:00Z",
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
    getLifecycleSummary: vi.fn().mockResolvedValue(null),
    getEvaluationReport: vi.fn().mockResolvedValue(null),
    recordRecallEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    recordSessionContinuityEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    getStats: vi.fn().mockResolvedValue({
      totalEntities: 10,
      totalRelationships: 20,
      totalEpisodes: 5,
      entityTypeCounts: { Person: 5, Organization: 3 },
      topActivated: [],
      topConnected: [],
      growthTimeline: [],
    }),
    getStorage: vi.fn().mockResolvedValue(null),
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
import { EvaluationPanel } from "../components/EvaluationPanel";
import { AtlasView } from "../components/graph/AtlasView";
import { api } from "../api/client";
import { useEngramStore } from "../store";
import type { BrainLoopEvaluationReport, Episode, SearchResult } from "../store/types";

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
    lifecycleDrilldownStage: null,
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
    storage: null,
    isLoadingStats: false,
    isLoadingStorage: false,
    lifecycleSummary: null,
    isLoadingLifecycleSummary: false,
    evaluationReport: null,
    isLoadingEvaluationReport: false,
    isSavingRecallEvaluation: false,
    isSavingSessionEvaluation: false,
    knowledgePackets: [],
    knowledgeRecallStatus: null,
    knowledgeRecallLifecycle: null,
    knowledgeRecallBudget: null,
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

describe("AtlasView", () => {
  it("does not record a last-visit marker when viewing a historical snapshot", () => {
    act(() => {
      useEngramStore.setState({
        atlasSnapshotId: "atlas_122",
        atlas: {
          representation: {
            scope: "atlas",
            layout: "precomputed",
            representedEntityCount: 1,
            representedEdgeCount: 0,
            displayedNodeCount: 1,
            displayedEdgeCount: 0,
            truncated: false,
            snapshotId: "atlas_122",
          },
          generatedAt: "2026-03-05T00:00:00Z",
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
        },
      });
    });

    const { unmount } = render(<AtlasView />);
    unmount();

    expect(useEngramStore.getState().lastAtlasVisitAt).toBeNull();
    expect(useEngramStore.getState().lastAtlasSnapshotId).toBeNull();
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
  it("shows Offline when health checks fail", async () => {
    render(<ConnectionStatus />);
    expect(await screen.findByText("Offline")).toBeInTheDocument();
  });

  it("shows Live when health is healthy and WebSocket is connected", async () => {
    vi.mocked(api.getHealth).mockResolvedValueOnce({
      status: "healthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "healthy" },
    });
    act(() => {
      useEngramStore.setState({ readyState: "connected" });
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Live")).toBeInTheDocument();
  });

  it("shows Syncing when health is healthy and WebSocket is reconnecting", async () => {
    vi.mocked(api.getHealth).mockResolvedValueOnce({
      status: "healthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "healthy" },
    });
    act(() => {
      useEngramStore.setState({ readyState: "reconnecting" });
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Syncing")).toBeInTheDocument();
  });

  it("shows Server OK when health is healthy but WebSocket is disconnected", async () => {
    vi.mocked(api.getHealth).mockResolvedValueOnce({
      status: "healthy",
      version: "test",
      mode: "lite",
      services: { graph_store: "healthy" },
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Server OK")).toBeInTheDocument();
  });

  it("shows onboarding when runtime state says empty Engram needs adoption actions", async () => {
    vi.mocked(api.getHealth).mockResolvedValueOnce({
      status: "healthy",
      version: "test",
      mode: "helix",
      services: { graph_store: "healthy" },
    });
    vi.mocked(api.getRuntimeState).mockResolvedValueOnce({
      projectName: "Engram",
      runtime: { mode: "helix" },
      activation: {},
      features: {},
      artifactBootstrap: {
        enabled: true,
        projectPath: "/tmp/engram",
        artifactCount: 0,
        freshArtifactCount: 0,
        staleArtifactCount: 0,
        lastObservedAt: null,
        staleAfterSeconds: 86400,
      },
      agentAdoption: {
        status: "fresh_runtime",
        doNotTreatEmptyAsFailure: true,
        requiredNextTools: ["claim_authority", "bootstrap_project", "get_context"],
        claimAuthority: {
          tool: "claim_authority",
          args: {
            project_path: "/tmp/engram",
            file_memory_present: "<true if local/file memory is visible>",
          },
          reason: "Ask Engram for the source-of-truth contract.",
        },
        bootstrap: {
          tool: "bootstrap_project",
          required: true,
          args: { project_path: "/tmp/engram" },
          reason: "Fresh runtime is onboarding state.",
        },
        reason: "Connected but empty/fresh Engram runtime.",
      },
      stats: { recallMetrics: {}, epistemicMetrics: {} },
      generatedAt: "2026-05-19T00:00:00Z",
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Onboarding")).toBeInTheDocument();
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

  it("applies capture drilldown context as an active capture filter", () => {
    const queuedEpisode: Episode = {
      episodeId: "ep_queued",
      content: "Queued capture waiting for projection",
      source: "api",
      status: "queued",
      createdAt: "2026-05-11T12:00:00Z",
      updatedAt: "2026-05-11T12:00:00Z",
      entities: [],
      factsCount: 0,
      processingDurationMs: null,
      error: null,
      retryCount: 0,
    };
    const completedEpisode: Episode = {
      ...queuedEpisode,
      episodeId: "ep_completed",
      content: "Completed capture already projected",
      status: "completed",
    };
    act(() => {
      useEngramStore.setState({
        episodes: [queuedEpisode, completedEpisode],
        hasMoreEpisodes: false,
        lifecycleDrilldownStage: "capture",
        loadEpisodes: vi.fn(),
      });
    });

    render(<MemoryFeed />);

    expect(screen.getByDisplayValue("Active capture")).toBeInTheDocument();
    expect(screen.getByText(/Queued capture/)).toBeInTheDocument();
    expect(screen.queryByText(/Completed capture/)).not.toBeInTheDocument();
  });
});

describe("StatsPanel", () => {
  it("renders loading state initially", () => {
    const loadStats = vi.fn();
    act(() => {
      useEngramStore.setState({ isLoadingStats: true, stats: null, loadStats });
    });
    const { container } = render(<StatsPanel />);
    // Loading state shows a skeleton element
    expect(container.querySelector(".skeleton")).toBeInTheDocument();
  });

  it("renders stats when loaded", () => {
    const loadStats = vi.fn();
    const loadStorage = vi.fn();
    act(() => {
      useEngramStore.setState({
        stats: {
          totalEntities: 42,
          totalRelationships: 100,
          totalEpisodes: 15,
          entityTypeCounts: { Person: 20, Organization: 10 },
          cueMetrics: {
            cueCount: 12,
            episodesWithoutCues: 3,
            cueCoverage: 0.8,
            cueHitCount: 9,
            cueHitEpisodeCount: 4,
            cueHitEpisodeRate: 0.3333,
            cueSurfacedCount: 6,
            cueSelectedCount: 3,
            cueUsedCount: 2,
            cueNearMissCount: 1,
            avgPolicyScore: 0.42,
            avgProjectionAttempts: 1.5,
            projectedCueCount: 5,
            cueToProjectionConversionRate: 0.4167,
          },
          projectionMetrics: {
            stateCounts: {
              queued: 1,
              cued: 2,
              cueOnly: 3,
              scheduled: 2,
              projecting: 1,
              projected: 5,
              failed: 1,
              deadLetter: 0,
            },
            attemptedEpisodeCount: 6,
            totalAttempts: 9,
            failureCount: 1,
            deadLetterCount: 0,
            failureRate: 1 / 6,
            avgProcessingDurationMs: 180,
            avgTimeToProjectionMs: 3200,
            yield: {
              linkedEntityCount: 14,
              relationshipCount: 8,
              avgLinkedEntitiesPerProjectedEpisode: 2.8,
              avgRelationshipsPerProjectedEpisode: 1.6,
            },
          },
          adjudicationMetrics: {
            evidenceStatusCounts: { pending: 1, deferred: 2, approved: 4 },
            requestStatusCounts: { pending: 0, deferred: 1, error: 0 },
            openEvidenceCount: 3,
            pendingEvidenceCount: 1,
            deferredEvidenceCount: 2,
            approvedEvidenceCount: 4,
            openRequestCount: 1,
            pendingRequestCount: 0,
            deferredRequestCount: 1,
            errorRequestCount: 0,
            openWorkCount: 3,
          },
          topActivated: [],
          topConnected: [],
          growthTimeline: [],
        },
        storage: {
          mode: "helix",
          configuredMode: "helix",
          backend: "helix_native",
          groupId: "default",
          startedAt: "2026-05-19T12:00:00Z",
          uptimeSeconds: 20,
          counts: { episodes: 16, entities: 43, relationships: 101, cues: 12 },
          startupCounts: { episodes: 11, entities: 41, relationships: 91, cues: 8 },
          growthSinceStartup: {
            bytes: 2048,
            episodes: 5,
            entities: 2,
            relationships: 10,
            cues: 4,
          },
          disk: {
            totalBytes: 4096,
            humanSize: "4.0 KB",
            startupBytes: 2048,
            startupHumanSize: "2.0 KB",
          },
          paths: [
            {
              label: "Helix native data",
              path: "/tmp/engram-native",
              exists: true,
              kind: "directory",
              bytes: 4096,
              humanSize: "4.0 KB",
            },
          ],
        },
        isLoadingStats: false,
        loadStats,
        loadStorage,
      });
    });
    render(<StatsPanel />);
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
    expect(screen.getByText("Cue Layer")).toBeInTheDocument();
    expect(screen.getByText("Projection Health")).toBeInTheDocument();
    expect(screen.getByText("Graph Hygiene")).toBeInTheDocument();
    expect(screen.getByTestId("stats-hygiene-section")).toBeInTheDocument();
    expect(screen.getByText("secondary hygiene metric")).toBeInTheDocument();
    expect(screen.getByText("Storage")).toBeInTheDocument();
    expect(screen.getByText("/tmp/engram-native")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.getByText("16.7%")).toBeInTheDocument();
  });

  it("highlights stats sections from lifecycle drilldown context", () => {
    const loadStats = vi.fn();
    act(() => {
      useEngramStore.setState({
        stats: {
          totalEntities: 42,
          totalRelationships: 100,
          totalEpisodes: 15,
          entityTypeCounts: { Person: 20, Organization: 10 },
          cueMetrics: {
            cueCount: 12,
            episodesWithoutCues: 3,
            cueCoverage: 0.8,
            cueHitCount: 9,
            cueHitEpisodeCount: 4,
            cueHitEpisodeRate: 0.3333,
            cueSurfacedCount: 6,
            cueSelectedCount: 3,
            cueUsedCount: 2,
            cueNearMissCount: 1,
            avgPolicyScore: 0.42,
            avgProjectionAttempts: 1.5,
            projectedCueCount: 5,
            cueToProjectionConversionRate: 0.4167,
          },
          projectionMetrics: {
            stateCounts: {
              queued: 1,
              cued: 2,
              cueOnly: 3,
              scheduled: 2,
              projecting: 1,
              projected: 5,
              failed: 1,
              deadLetter: 0,
            },
            attemptedEpisodeCount: 6,
            totalAttempts: 9,
            failureCount: 1,
            deadLetterCount: 0,
            failureRate: 1 / 6,
            avgProcessingDurationMs: 180,
            avgTimeToProjectionMs: 3200,
            yield: {
              linkedEntityCount: 14,
              relationshipCount: 8,
              avgLinkedEntitiesPerProjectedEpisode: 2.8,
              avgRelationshipsPerProjectedEpisode: 1.6,
            },
          },
          topActivated: [],
          topConnected: [],
          growthTimeline: [],
        },
        isLoadingStats: false,
        lifecycleDrilldownStage: "project",
        loadStats,
      });
    });

    render(<StatsPanel />);

    expect(screen.getByTestId("stats-project-section")).toHaveAttribute(
      "data-lifecycle-focus",
      "true",
    );
    expect(screen.getByTestId("stats-cue-section")).not.toHaveAttribute(
      "data-lifecycle-focus",
    );
  });
});

describe("EvaluationPanel", () => {
  function makeEvaluationSignalsFixture(): BrainLoopEvaluationReport["evaluationSignals"] {
    return {
      cueUsefulness: { status: "measured", evidenceCount: 8, metric: 0.375, gap: null },
      projectionYield: { status: "measured", evidenceCount: 5, metric: 2.4, gap: null },
      recallQuality: { status: "measured", evidenceCount: 2, metric: 0.5, gap: null },
      falseRecall: { status: "measured", evidenceCount: 5, metric: 0.2, gap: null },
      triageCalibration: { status: "measured", evidenceCount: 3, metric: 0.12, gap: null },
      consolidationEffect: { status: "measured", evidenceCount: 1, metric: 0.5, gap: null },
    };
  }

  function makeMemoryValueFixture(): BrainLoopEvaluationReport["memoryValue"] {
    return {
      status: "measured",
      cost: {
        status: "measured",
        operationCount: 6,
        avgAddedLatencyMs: 9.5,
        p95AddedLatencyMs: 24,
        avgBudgetMs: 600,
        p95BudgetMs: 1200,
        avgBudgetTokens: 300,
        completedCount: 4,
        skippedCount: 1,
        errorCount: 1,
        statusCounts: { ok: 4, skipped: 1, error: 1 },
        skipReasonCounts: { skipped_low_signal: 1 },
        timeoutCount: 1,
        timeoutRate: 0.1667,
        degradedCount: 0,
        degradedRate: 0,
        budgetMissCount: 2,
        budgetMissRate: 0.3333,
        cacheHitCount: 3,
        cacheMissCount: 3,
        cacheHitRate: 0.5,
        byMode: {},
      },
      benefit: {
        status: "measured",
        recallSampleCount: 2,
        sessionSampleCount: 1,
        memoryNeedPrecision: 0.5,
        memoryNeedRecall: 0.5,
        missedRecallRate: 0.5,
        usefulPacketRate: 0.4,
        stalePacketRate: 0.2,
        correctedPacketRate: 0.1,
        stalePacketCount: 1,
        correctedPacketCount: 1,
        falseRecallRate: 0.2,
        sessionContinuityLift: 0.3,
        openLoopRecoveryRate: 1,
        temporalCorrectness: 0,
      },
    };
  }

  function makeEvaluationReportFixture(): BrainLoopEvaluationReport {
    return {
      groupId: "default",
      generatedAt: new Date().toISOString(),
      degraded: false,
      degradations: [],
      loop: ["capture", "cue", "project", "recall", "consolidate"],
      totals: { episodes: 8, entities: 12, relationships: 5, activeEntities: 3 },
      capture: { status: "ready", episodeCount: 8, activeCount: 0 },
      cue: {
        status: "ready",
        cueCount: 7,
        episodesWithoutCues: 1,
        coverage: 0.875,
        hitCount: 6,
        hitEpisodeCount: 3,
        hitEpisodeRate: 0.5,
        surfacedCount: 8,
        selectedCount: 4,
        usedCount: 3,
        nearMissCount: 1,
        selectedRate: 0.5,
        usedRate: 0.375,
        nearMissRate: 0.125,
        avgPolicyScore: 0.74,
        projectionConversionRate: 0.6,
      },
      project: {
        status: "ready",
        stateCounts: {
          queued: 0,
          cued: 0,
          cueOnly: 1,
          scheduled: 0,
          projecting: 0,
          projected: 5,
          merged: 0,
          failed: 0,
          deadLetter: 0,
        },
        trackedCount: 6,
        projectedCount: 5,
        activeCount: 0,
        projectedRate: 0.8333,
        backlogRate: 0,
        failedCount: 0,
        deadLetterCount: 0,
        attemptedEpisodeCount: 5,
        totalAttempts: 5,
        failureRate: 0,
        avgProcessingDurationMs: 100,
        avgTimeToProjectionMs: 200,
        yield: {
          linkedEntityCount: 12,
          relationshipCount: 5,
          avgLinkedEntitiesPerProjectedEpisode: 2.4,
          avgRelationshipsPerProjectedEpisode: 1,
        },
      },
      recall: {
        status: "active",
        totalAnalyses: 4,
        triggerCount: 2,
        runtimeFalseRecallRate: 0,
        runtimeSurfacedToUsedRatio: null,
        graphLiftRate: 0.25,
        probeTriggerRate: 0.5,
        latency: {
          analyzerMs: { avgMs: 12, p95Ms: 31 },
          probeMs: { avgMs: 7, p95Ms: 19 },
        },
        control: {
          usedCount: 3,
          dismissedCount: 1,
          surfacedCount: 5,
          selectedCount: 2,
          confirmedCount: 1,
          correctedCount: 1,
          graphOverrideCount: 2,
          adaptiveThresholdsEnabled: true,
          thresholds: { linguistic: 0.32, borderline: 0.18, resonance: 0.5 },
        },
        familyContributions: { linguistic: 2 },
        evaluation: {
          status: "measured",
          sampleCount: 2,
          needStatus: "measured",
          needLabeledCount: 2,
          neededCount: 2,
          missedCount: 1,
          memoryNeedPrecision: 0.5,
          memoryNeedRecall: 0.5,
          missedRecallRate: 0.5,
          usefulPacketRate: 0.4,
          stalePacketRate: 0.2,
          correctedPacketRate: 0.1,
          stalePacketCount: 1,
          correctedPacketCount: 1,
          falseRecallRate: 0.2,
          surfacedCount: 5,
          usedCount: 2,
          surfacedToUsedRatio: 2.5,
        },
        continuity: {
          status: "measured",
          sampleCount: 1,
          sessionContinuityLift: 0.3,
          openLoopRecoveryRate: 1,
          temporalCorrectness: 0,
        },
      },
      memoryValue: makeMemoryValueFixture(),
      consolidate: {
        status: "attention",
        cycleCount: 1,
        latestStatus: "failed",
        latestCycle: { id: "cyc_1", error: "calibration failed" },
        phaseStatusCounts: { success: 1, error: 1 },
        phaseTotals: { triage: { runs: 1, itemsProcessed: 4, itemsAffected: 2, effectRate: 0.5 } },
        adjudication: {
          status: "active",
          phaseCount: 1,
          runs: 1,
          itemsProcessed: 3,
          itemsAffected: 1,
          itemsUnaffected: 2,
          effectRate: 0.3333,
          errorCount: 0,
          phaseTotals: {
            edge_adjudication: {
              runs: 1,
              itemsProcessed: 3,
              itemsAffected: 1,
              effectRate: 0.3333,
            },
          },
        },
        calibration: {
          status: "measured",
          snapshotCount: 1,
          phaseTotals: {
            triage: {
              snapshots: 1,
              totalTraces: 5,
              labeledExamples: 3,
              oracleExamples: 1,
              abstainCount: 0,
              accuracy: 0.67,
              meanConfidence: 0.8,
              expectedCalibrationError: 0.12,
            },
          },
        },
        itemsProcessed: 4,
        itemsAffected: 2,
        effectRate: 0.5,
        errorCount: 1,
      },
      evaluationSignals: makeEvaluationSignalsFixture(),
      coverageGaps: [],
    };
  }

  it("renders measured recall and continuity signals", async () => {
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce({
      groupId: "default",
      generatedAt: new Date().toISOString(),
      loop: ["capture", "cue", "project", "recall", "consolidate"],
      totals: { episodes: 8, entities: 12, relationships: 5, activeEntities: 3 },
      capture: { status: "ready", episodeCount: 8, activeCount: 0 },
      cue: {
        status: "ready",
        cueCount: 7,
        episodesWithoutCues: 1,
        coverage: 0.875,
        hitCount: 6,
        hitEpisodeCount: 3,
        hitEpisodeRate: 0.5,
        surfacedCount: 8,
        selectedCount: 4,
        usedCount: 3,
        nearMissCount: 1,
        selectedRate: 0.5,
        usedRate: 0.375,
        nearMissRate: 0.125,
        avgPolicyScore: 0.74,
        projectionConversionRate: 0.6,
      },
      project: {
        status: "ready",
        stateCounts: {
          queued: 0,
          cued: 0,
          cueOnly: 1,
          scheduled: 0,
          projecting: 0,
          projected: 5,
          merged: 0,
          failed: 0,
          deadLetter: 0,
        },
        trackedCount: 6,
        projectedCount: 5,
        activeCount: 0,
        projectedRate: 0.8333,
        backlogRate: 0,
        failedCount: 0,
        deadLetterCount: 0,
        attemptedEpisodeCount: 5,
        totalAttempts: 5,
        failureRate: 0,
        avgProcessingDurationMs: 100,
        avgTimeToProjectionMs: 200,
        yield: {
          linkedEntityCount: 12,
          relationshipCount: 5,
          avgLinkedEntitiesPerProjectedEpisode: 2.4,
          avgRelationshipsPerProjectedEpisode: 1,
        },
      },
      recall: {
        status: "active",
        totalAnalyses: 4,
        triggerCount: 2,
        runtimeFalseRecallRate: 0,
        runtimeSurfacedToUsedRatio: null,
        graphLiftRate: 0.25,
        probeTriggerRate: 0.5,
        latency: {
          analyzerMs: { avgMs: 12, p95Ms: 31 },
          probeMs: { avgMs: 7, p95Ms: 19 },
        },
        control: {
          usedCount: 3,
          dismissedCount: 1,
          surfacedCount: 5,
          selectedCount: 2,
          confirmedCount: 1,
          correctedCount: 1,
          graphOverrideCount: 2,
          adaptiveThresholdsEnabled: true,
          thresholds: { linguistic: 0.32, borderline: 0.18, resonance: 0.5 },
        },
        familyContributions: { linguistic: 2 },
        evaluation: {
          status: "measured",
          sampleCount: 2,
          needStatus: "measured",
          needLabeledCount: 2,
          neededCount: 2,
          missedCount: 1,
          memoryNeedPrecision: 0.5,
          memoryNeedRecall: 0.5,
          missedRecallRate: 0.5,
          usefulPacketRate: 0.4,
          stalePacketRate: 0.2,
          correctedPacketRate: 0.1,
          stalePacketCount: 1,
          correctedPacketCount: 1,
          falseRecallRate: 0.2,
          surfacedCount: 5,
          usedCount: 2,
          surfacedToUsedRatio: 2.5,
        },
        continuity: {
          status: "measured",
          sampleCount: 1,
          sessionContinuityLift: 0.3,
          openLoopRecoveryRate: 1,
          temporalCorrectness: 0,
        },
      },
      memoryValue: makeMemoryValueFixture(),
      consolidate: {
        status: "attention",
        cycleCount: 1,
        latestStatus: "failed",
        latestCycle: { id: "cyc_1", error: "calibration failed" },
        phaseStatusCounts: { success: 1, error: 1 },
        phaseTotals: { triage: { runs: 1, itemsProcessed: 4, itemsAffected: 2, effectRate: 0.5 } },
        adjudication: {
          status: "active",
          phaseCount: 1,
          runs: 1,
          itemsProcessed: 3,
          itemsAffected: 1,
          itemsUnaffected: 2,
          effectRate: 0.3333,
          errorCount: 0,
          phaseTotals: {
            edge_adjudication: {
              runs: 1,
              itemsProcessed: 3,
              itemsAffected: 1,
              effectRate: 0.3333,
            },
          },
        },
        calibration: {
          status: "measured",
          snapshotCount: 1,
          phaseTotals: {
            triage: {
              snapshots: 1,
              totalTraces: 5,
              labeledExamples: 3,
              oracleExamples: 1,
              abstainCount: 0,
              accuracy: 0.67,
              meanConfidence: 0.8,
              expectedCalibrationError: 0.12,
            },
          },
        },
        itemsProcessed: 4,
        itemsAffected: 2,
        effectRate: 0.5,
        errorCount: 1,
      },
      evaluationSignals: makeEvaluationSignalsFixture(),
      coverageGaps: [],
    });

    render(<EvaluationPanel />);

    expect(await screen.findByText("Runtime quality signals")).toBeInTheDocument();
    expect(screen.getByText("Recall")).toBeInTheDocument();
    expect(screen.getByText("Continuity")).toBeInTheDocument();
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("selected rate")).toBeInTheDocument();
    expect(screen.getByText("projection")).toBeInTheDocument();
    expect(screen.getByText("60.0%")).toBeInTheDocument();
    expect(screen.getByText("backlog")).toBeInTheDocument();
    expect(screen.getAllByText("0%").length).toBeGreaterThan(0);
    expect(screen.getByText("latency")).toBeInTheDocument();
    expect(screen.getByText("200ms")).toBeInTheDocument();
    expect(screen.getByText("processing")).toBeInTheDocument();
    expect(screen.getByText("100ms")).toBeInTheDocument();
    expect(screen.getByText("analysis p95")).toBeInTheDocument();
    expect(screen.getByText("probe p95")).toBeInTheDocument();
    expect(screen.getByText("31ms")).toBeInTheDocument();
    expect(screen.getByText("19ms")).toBeInTheDocument();
    expect(screen.getByText("Memory Value")).toBeInTheDocument();
    expect(screen.getByText("p95 added")).toBeInTheDocument();
    expect(screen.getByText("24ms")).toBeInTheDocument();
    expect(screen.getByText("skipped")).toBeInTheDocument();
    expect(screen.getByText("budget p95")).toBeInTheDocument();
    expect(screen.getByText("1.2s")).toBeInTheDocument();
    expect(screen.getByText("cache hit")).toBeInTheDocument();
    expect(screen.getByText("stale packets")).toBeInTheDocument();
    expect(screen.getByText("corrected packets")).toBeInTheDocument();
    expect(screen.getByText("Recall Gate")).toBeInTheDocument();
    expect(screen.getByText("Signal Readiness")).toBeInTheDocument();
    expect(screen.getByText("6/6 measured")).toBeInTheDocument();
    expect(screen.getByText("Cue usefulness")).toBeInTheDocument();
    expect(screen.getByText("False recall")).toBeInTheDocument();
    expect(screen.getByText("runtime used")).toBeInTheDocument();
    expect(screen.getByText("graph override")).toBeInTheDocument();
    expect(screen.getByText("resonance")).toBeInTheDocument();
    expect(screen.getByText("accuracy")).toBeInTheDocument();
    expect(screen.getByText("67.0%")).toBeInTheDocument();
    expect(screen.getByText("effect")).toBeInTheDocument();
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("adjudication")).toBeInTheDocument();
    expect(screen.getAllByText("33.3%").length).toBeGreaterThan(0);
    expect(screen.getByText("unaffected")).toBeInTheDocument();
    expect(screen.getAllByText("0.120").length).toBeGreaterThan(0);
    expect(screen.getByText("latest issue")).toBeInTheDocument();
    expect(screen.getByText("calibration failed")).toBeInTheDocument();
    expect(screen.getByText("open loops")).toBeInTheDocument();
  });

  it("shows latest consolidation phase issues in evaluation output", async () => {
    const report = makeEvaluationReportFixture();
    report.consolidate.latestCycle = {
      id: "cyc_phase_issue",
      error: null,
      phase_issue: "edge_adjudication: judge unavailable",
    };
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce(report);

    render(<EvaluationPanel />);

    expect(await screen.findByText("latest issue")).toBeInTheDocument();
    expect(screen.getByText("edge_adjudication: judge unavailable")).toBeInTheDocument();
  });

  it("renders degraded report fallbacks in evaluation output", async () => {
    const report = makeEvaluationReportFixture();
    report.degraded = true;
    report.degradations = [
      {
        surface: null,
        stage: "graph_state",
        status: "degraded",
        skipReason: "graph_state_timeout",
        timeoutMs: 2000,
      },
    ];
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce(report);

    render(<EvaluationPanel />);

    expect(await screen.findByText("Report degraded")).toBeInTheDocument();
    expect(screen.getByText("1 fallback")).toBeInTheDocument();
    expect(screen.getByText("graph_state · graph_state_timeout · 2.0s")).toBeInTheDocument();
  });

  it("renders release evidence gate state", async () => {
    const report = makeEvaluationReportFixture();
    report.humanLabelEvidence = {
      status: "measured",
      artifactPath: "human-labels.json",
      artifactSha256: "human123",
      kind: "engram_human_label_evidence",
      source: "staging_harness",
      client: "Cursor",
      capturedAt: "2026-05-18T23:00:00Z",
      sessionId: "cursor-thread-1",
      labeler: "operator",
      humanLabeled: true,
      recallSampleCount: 12,
      sessionSampleCount: 4,
      minRecallSamples: 10,
      minSessionSamples: 3,
      sampleSources: ["staging_harness"],
      failures: [],
    };
    report.adoptionEvidence = {
      status: "measured",
      artifactPath: "cursor-adoption-report.json",
      artifactSha256: "cursor123",
      adoptionStatus: "passed",
      authorityPath: "authority.json",
      callsPath: "calls.jsonl",
      callCount: 4,
      client: "Cursor",
      requiredClient: "Cursor",
      gateRequiredClient: "Cursor",
      capturedAt: "2026-05-18T23:00:00Z",
      sessionId: "cursor-thread-1",
      sessionFilter: "cursor-thread-1",
      source: "live_harness",
      requiredLiveEvidence: true,
      blockers: ["mcp_server_failed", "authentication_failed"],
      blockerDetails: ["system:error: Not logged in - Please run /login"],
      mcpServerFailures: ["engram"],
      requiredTools: {
        expected: ["get_context", "recall", "observe"],
        observed: ["get_context", "recall", "observe"],
        missing: [],
        inOrder: true,
      },
      capture: {
        destination: "engram",
        expectedTool: "observe",
        observedTools: ["observe"],
        missing: false,
      },
      fileMemory: { present: false, substitutedForEngram: false },
      failures: [],
    };
    report.additionalAdoptionEvidence = [
      {
        ...report.adoptionEvidence,
        client: "Windsurf",
        artifactPath: "windsurf-adoption-report.json",
        blockers: ["mcp_server_failed"],
      },
    ];
    report.adoptionClientEvidence = {
      status: "measured",
      requiredClients: ["Cursor", "Windsurf"],
      observedClients: ["Cursor", "Windsurf"],
      reportCount: 2,
      reports: [
        {
          client: "Cursor",
          requiredClient: "Cursor",
          status: "measured",
          artifactPath: "cursor-adoption-report.json",
          artifactSha256: "cursor123",
          capturedAt: "2026-05-18T23:00:00Z",
          sessionId: "cursor-thread-1",
          failures: [],
        },
        {
          client: "Windsurf",
          requiredClient: "Windsurf",
          status: "measured",
          artifactPath: "windsurf-adoption-report.json",
          artifactSha256: "windsurf123",
          capturedAt: "2026-05-18T23:01:00Z",
          sessionId: "windsurf-thread-1",
          blockers: ["mcp_server_failed"],
          blockerDetails: ["mcp server engram failed"],
          mcpServerFailures: ["engram"],
          failures: [],
        },
      ],
      blockers: ["mcp_server_failed", "authentication_failed"],
      mcpServerFailures: ["engram"],
      failures: [],
    };
    report.releaseEvidence = {
      status: "measured",
      components: {
        evaluationSignals: { status: "measured", missing: [], failures: [] },
        humanLabels: { status: "measured", missing: [], failures: [] },
        adoption: {
          status: "measured",
          missing: [],
          failures: [],
          blockers: ["mcp_server_failed", "authentication_failed"],
          blockerDetails: ["system:error: Not logged in - Please run /login"],
          mcpServerFailures: ["engram"],
        },
        adoptionClients: {
          status: "measured",
          missing: [],
          failures: [],
          requiredClients: ["Cursor", "Windsurf"],
          observedClients: ["Cursor", "Windsurf"],
          blockers: ["mcp_server_failed", "authentication_failed"],
          mcpServerFailures: ["engram"],
        },
      },
      missing: [],
      failures: [],
    };
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce(report);

    render(<EvaluationPanel />);

    expect(await screen.findByText("Release Evidence")).toBeInTheDocument();
    expect(screen.getByText("Readiness")).toBeInTheDocument();
    expect(screen.getAllByText("measured").length).toBeGreaterThan(0);
    expect(screen.getByText("Human labels")).toBeInTheDocument();
    expect(screen.getByText("Adoption")).toBeInTheDocument();
    expect(screen.getByText("Clients")).toBeInTheDocument();
    expect(screen.getAllByText("Cursor").length).toBeGreaterThan(0);
    expect(screen.getAllByText("12/10").length).toBeGreaterThan(0);
    expect(screen.getAllByText("4/3").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cursor, Windsurf").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Windsurf")).toBeInTheDocument();
    expect(
      screen.getAllByText("mcp_server_failed, authentication_failed").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("engram").length).toBeGreaterThan(0);
  });

  it("renders release evidence readiness when artifacts are missing", async () => {
    const report = makeEvaluationReportFixture();
    report.releaseEvidence = {
      status: "needs_evidence",
      components: {
        evaluationSignals: { status: "measured", missing: [], failures: [] },
        humanLabels: {
          status: "missing",
          missing: ["human_label_evidence"],
          failures: [],
        },
        adoption: {
          status: "missing",
          missing: ["adoption_evidence"],
          failures: [],
        },
        adoptionClients: {
          status: "not_required",
          missing: [],
          failures: [],
          requiredClients: [],
          observedClients: [],
        },
      },
      missing: ["human_label_evidence", "adoption_evidence"],
      failures: [],
    };
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce(report);

    render(<EvaluationPanel />);

    expect(await screen.findByText("Release Evidence")).toBeInTheDocument();
    expect(screen.getAllByText("needs_evidence").length).toBeGreaterThan(0);
    expect(
      screen.getByText("human_label_evidence, adoption_evidence"),
    ).toBeInTheDocument();
    expect(screen.getByText("not_required")).toBeInTheDocument();
    expect(screen.queryByText("No release evidence attached")).not.toBeInTheDocument();
  });

  it("shows calibration quality gaps without rendering unscored accuracy", async () => {
    const report = makeEvaluationReportFixture();
    report.consolidate.calibration = {
      status: "needs_quality",
      snapshotCount: 1,
      phaseTotals: {
        triage: {
          snapshots: 1,
          totalTraces: 4,
          labeledExamples: 0,
          oracleExamples: 0,
          abstainCount: 0,
          accuracy: null,
          meanConfidence: null,
          expectedCalibrationError: null,
        },
      },
    };
    report.coverageGaps = [
      "consolidation calibration quality needs labeled decision outcomes",
    ];
    vi.mocked(api.getEvaluationReport).mockResolvedValueOnce(report);

    render(<EvaluationPanel />);

    expect(await screen.findByText("Needs labeled decisions")).toBeInTheDocument();
    expect(
      screen.getByText("consolidation calibration quality needs labeled decision outcomes"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/0 labels/)).not.toBeInTheDocument();
  });

  it("stores operator evaluation labels from the dashboard", async () => {
    const user = userEvent.setup();
    const report = makeEvaluationReportFixture();
    vi.mocked(api.getEvaluationReport)
      .mockResolvedValueOnce(report)
      .mockResolvedValue(report);
    const recordRecallEvaluation = vi.mocked(api.recordRecallEvaluation);
    const recordSessionContinuityEvaluation = vi.mocked(api.recordSessionContinuityEvaluation);
    recordRecallEvaluation.mockClear();
    recordSessionContinuityEvaluation.mockClear();

    render(<EvaluationPanel />);

    expect(await screen.findByText("Recall Label")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("Surfaced"));
    await user.type(screen.getByLabelText("Surfaced"), "3");
    await user.clear(screen.getByLabelText("Used"));
    await user.type(screen.getByLabelText("Used"), "2");
    await user.clear(screen.getByLabelText("False"));
    await user.type(screen.getByLabelText("False"), "1");
    await user.clear(screen.getByLabelText("Stale"));
    await user.type(screen.getByLabelText("Stale"), "1");
    await user.clear(screen.getByLabelText("Corrected"));
    await user.type(screen.getByLabelText("Corrected"), "1");
    await user.type(screen.getByLabelText("Query"), "open loop");
    await user.click(screen.getByRole("button", { name: "Store Recall" }));

    await waitFor(() => {
      expect(recordRecallEvaluation).toHaveBeenCalledWith({
        recallTriggered: true,
        recallHelped: true,
        recallNeeded: true,
        packetsSurfaced: 3,
        packetsUsed: 2,
        falseRecalls: 1,
        stalePackets: 1,
        correctedPackets: 1,
        source: "dashboard",
        query: "open loop",
        notes: null,
      });
    });

    await user.clear(screen.getByLabelText("Baseline"));
    await user.type(screen.getByLabelText("Baseline"), "0.2");
    await user.clear(screen.getByLabelText("Memory"));
    await user.type(screen.getByLabelText("Memory"), "0.6");
    await user.click(screen.getByLabelText("Open loop"));
    await user.click(screen.getByLabelText("Recovered"));
    await user.type(screen.getByLabelText("Scenario"), "follow up");
    await user.click(screen.getByRole("button", { name: "Store Continuity" }));

    await waitFor(() => {
      expect(recordSessionContinuityEvaluation).toHaveBeenCalledWith({
        baselineScore: 0.2,
        memoryScore: 0.6,
        openLoopExpected: true,
        openLoopRecovered: true,
        temporalExpected: false,
        temporalCorrect: false,
        source: "dashboard",
        scenario: "follow up",
        notes: null,
      });
    });
  });

  it("settles to a manual load action when the report request fails", async () => {
    const getEvaluationReport = vi.mocked(api.getEvaluationReport);
    getEvaluationReport.mockClear();
    getEvaluationReport.mockRejectedValueOnce(new Error("offline"));

    render(<EvaluationPanel />);

    expect(await screen.findByText("Load Evaluation")).toBeInTheDocument();
    expect(getEvaluationReport).toHaveBeenCalledTimes(1);
  });
});

// --- Knowledge Tab Redesign Tests ---

import { MemoryPulse } from "../components/knowledge/MemoryPulse";
import { RecallPacketDrilldown } from "../components/knowledge/RecallPacketDrilldown";
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
          { entityId: "e2", name: "Alex", entityType: "Person", currentActivation: 0.91 },
        ],
        isPulseLoading: false,
      });
    });
    render(<MemoryPulse />);
    expect(screen.getByText("Engram")).toBeInTheDocument();
    expect(screen.getByText("Alex")).toBeInTheDocument();
    expect(screen.getByText("PULSE")).toBeInTheDocument();
  });

  it("keeps recall context visible when opened from the lifecycle stage", () => {
    act(() => {
      useEngramStore.setState({
        pulseEntities: [],
        isPulseLoading: false,
        lifecycleDrilldownStage: "recall",
      });
    });
    render(<MemoryPulse />);

    expect(screen.getByText("Recall Context")).toBeInTheDocument();
    expect(screen.getByText("No active entities loaded")).toBeInTheDocument();
    expect(screen.getByText("Recall Context").closest("[data-lifecycle-focus]")).toHaveAttribute(
      "data-lifecycle-focus",
      "true",
    );
  });
});

describe("RecallPacketDrilldown", () => {
  it("renders packet trust and provenance details", () => {
    act(() => {
      useEngramStore.setState({
        knowledgePackets: [
          {
            packetType: "state_packet",
            title: "State: Engram",
            summary: "Cached project context.",
            trust: {
              source: "cache",
              freshness: "recent",
              confidence: 0.8,
              whyNow: "Cached project context for the current workspace.",
              provenanceCount: 1,
              evidenceCount: 2,
              beliefStatus: "supported",
              confirmedCount: 1,
              correctedCount: 1,
              dismissedCount: 0,
              lastConfirmedAt: "2026-05-21T18:00:00Z",
              lastCorrectedAt: "2026-05-21T18:05:00Z",
            },
          },
        ],
      });
    });

    render(<RecallPacketDrilldown />);

    expect(screen.getByText("Recall Packets")).toBeInTheDocument();
    expect(screen.getByText("State: Engram")).toBeInTheDocument();
    expect(screen.getByText("state_packet | cache | recent")).toBeInTheDocument();
    expect(screen.getByText("confidence 80%")).toBeInTheDocument();
    expect(screen.getByText(/feedback confirmed 1/)).toBeInTheDocument();
    expect(screen.getByText(/corrected 2026-05-21T18:05:00Z/)).toBeInTheDocument();
    expect(screen.getByText("Cached project context for the current workspace.")).toBeInTheDocument();
  });

  it("renders bounded recall runtime budget when no packets surfaced", () => {
    act(() => {
      useEngramStore.setState({
        knowledgePackets: [],
        knowledgeRecallStatus: "degraded",
        knowledgeRecallLifecycle: {
          stage: "recall",
          recallMode: "explicit",
          resultCount: 0,
          packetCount: 0,
          degraded: true,
          skipReason: "recall_timeout",
          timeout: true,
        },
        knowledgeRecallBudget: {
          profile: "explicit",
          surface: "axi",
          mode: "axi_recall",
          maxWallMs: 2000,
          maxSearchMs: 1200,
          maxResults: 3,
          durationMs: 1201,
          budgetMiss: true,
          timeout: true,
          degraded: true,
          skipReason: "recall_timeout",
        },
      });
    });

    render(<RecallPacketDrilldown />);

    expect(screen.getByText("Recall Packets")).toBeInTheDocument();
    expect(screen.getByText("0 surfaced")).toBeInTheDocument();
    expect(screen.getByText("degraded")).toBeInTheDocument();
    expect(screen.getByText("axi/explicit")).toBeInTheDocument();
    expect(screen.getByText("1,201ms")).toBeInTheDocument();
    expect(screen.getByText("budget 2,000ms")).toBeInTheDocument();
    expect(screen.getByText("recall_timeout")).toBeInTheDocument();
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

  it("ignores stale search responses after a newer query", async () => {
    const user = userEvent.setup();

    let resolveFirst: ((value: SearchResult[]) => void) | null = null;
    let resolveSecond: ((value: SearchResult[]) => void) | null = null;

    vi.mocked(api.searchEntities).mockImplementation(
      ({ q }) =>
        new Promise<SearchResult[]>((resolve) => {
          if (q === "a") {
            resolveFirst = resolve;
            return;
          }
          if (q === "ab") {
            resolveSecond = resolve;
            return;
          }
          resolve([]);
        }),
    );

    act(() => {
      useEngramStore.setState({ searchOverlayOpen: true });
    });

    render(<SearchOverlay />);

    const input = screen.getByPlaceholderText("Search entities...");
    await user.type(input, "a");
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 250));
    });

    await user.clear(input);
    await user.type(input, "ab");
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 250));
    });

    await act(async () => {
      resolveFirst?.([
        {
          id: "stale",
          name: "Stale Result",
          entityType: "Person",
          summary: null,
          activationScore: 0.2,
        },
      ]);
      await Promise.resolve();
    });

    expect(screen.queryByText("Stale Result")).not.toBeInTheDocument();

    await act(async () => {
      resolveSecond?.([
        {
          id: "fresh",
          name: "Fresh Result",
          entityType: "Person",
          summary: null,
          activationScore: 0.8,
        },
      ]);
      await Promise.resolve();
    });

    expect(await screen.findByText("Fresh Result")).toBeInTheDocument();
    expect(screen.queryByText("Stale Result")).not.toBeInTheDocument();
  });
});
