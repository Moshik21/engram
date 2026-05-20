import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LifecyclePanel } from "../components/LifecyclePanel";
import { api } from "../api/client";
import { useEngramStore } from "../store";
import type { ConsolidationCycleSummary, Episode, GraphStats, LifecycleSummary } from "../store/types";

vi.mock("../api/client", () => ({
  api: {
    getLifecycleSummary: vi.fn(),
    getEvaluationReport: vi.fn().mockResolvedValue(null),
    recordRecallEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    recordSessionContinuityEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    getStats: vi.fn().mockResolvedValue({
      totalEntities: 0,
      totalRelationships: 0,
      totalEpisodes: 0,
      entityTypeCounts: {},
      topActivated: [],
      topConnected: [],
      growthTimeline: [],
    }),
    getStorage: vi.fn().mockResolvedValue(null),
    getEpisodes: vi.fn().mockResolvedValue({ items: [], nextCursor: null }),
    getConsolidationStatus: vi.fn().mockResolvedValue({
      is_running: false,
      scheduler_active: false,
      pressure: null,
    }),
    getConsolidationHistory: vi.fn().mockResolvedValue({ cycles: [] }),
    getConsolidationCycle: vi.fn().mockResolvedValue({ cycle: null }),
  },
}));

const stats: GraphStats = {
  totalEntities: 42,
  totalRelationships: 88,
  totalEpisodes: 12,
  entityTypeCounts: { Person: 12 },
  cueMetrics: {
    cueCount: 10,
    episodesWithoutCues: 2,
    cueCoverage: 0.82,
    cueHitCount: 18,
    cueHitEpisodeCount: 6,
    cueHitEpisodeRate: 0.5,
    cueSurfacedCount: 20,
    cueSelectedCount: 9,
    cueUsedCount: 7,
    cueNearMissCount: 1,
    avgPolicyScore: 0.73,
    avgProjectionAttempts: 1.2,
    projectedCueCount: 8,
    cueToProjectionConversionRate: 0.8,
  },
  projectionMetrics: {
    stateCounts: {
      queued: 1,
      cued: 2,
      cueOnly: 1,
      scheduled: 3,
      projecting: 1,
      projected: 8,
      failed: 1,
      deadLetter: 0,
    },
    attemptedEpisodeCount: 9,
    totalAttempts: 11,
    failureCount: 1,
    deadLetterCount: 0,
    failureRate: 0.09,
    avgProcessingDurationMs: 120,
    avgTimeToProjectionMs: 240,
    yield: {
      linkedEntityCount: 21,
      relationshipCount: 17,
      avgLinkedEntitiesPerProjectedEpisode: 2.6,
      avgRelationshipsPerProjectedEpisode: 2.1,
    },
  },
  topActivated: [
    { id: "ent_a", name: "Alice", entityType: "Person", activation: 0.91 },
  ],
  topConnected: [],
  growthTimeline: [],
};

const episode: Episode = {
  episodeId: "ep_1",
  content: "Alice discussed the launch plan with the team",
  source: "api",
  status: "completed",
  projectionState: "projected",
  lastProjectionReason: "high_confidence",
  lastProjectedAt: "2026-05-10T12:00:00Z",
  conversationDate: null,
  createdAt: "2026-05-10T12:00:00Z",
  updatedAt: "2026-05-10T12:00:00Z",
  entities: [{ id: "ent_a", name: "Alice", entityType: "Person" }],
  factsCount: 2,
  processingDurationMs: 120,
  error: null,
  retryCount: 0,
  cue: {
    cueText: "launch plan",
    projectionState: "projected",
    routeReason: "salient",
    hitCount: 3,
    surfacedCount: 4,
    selectedCount: 2,
    usedCount: 1,
    nearMissCount: 0,
    policyScore: 0.74,
    projectionAttempts: 1,
    lastHitAt: "2026-05-10T12:30:00Z",
    lastFeedbackAt: null,
    lastProjectedAt: "2026-05-10T12:00:00Z",
  },
};

const cycle: ConsolidationCycleSummary = {
  id: "cyc_1",
  status: "failed",
  error: "calibration failed",
  dry_run: false,
  trigger: "manual",
  started_at: 1_779_000_000,
  completed_at: 1_779_000_003,
  total_duration_ms: 3000,
  phases: [
    {
      phase: "triage",
      status: "success",
      items_processed: 3,
      items_affected: 2,
      duration_ms: 20,
      error: null,
    },
    {
      phase: "merge",
      status: "success",
      items_processed: 2,
      items_affected: 1,
      duration_ms: 40,
      error: null,
    },
    {
      phase: "dream",
      status: "skipped",
      items_processed: 0,
      items_affected: 0,
      duration_ms: 0,
      error: null,
    },
  ],
};

function buildLifecycleSummary(overrides: Partial<LifecycleSummary> = {}): LifecycleSummary {
  return {
    groupId: "default",
    generatedAt: "2026-05-11T12:00:00Z",
    loop: ["capture", "cue", "project", "recall", "consolidate"],
    totals: {
      episodes: 12,
      cues: 10,
      projected: 8,
      cycles: 1,
      entities: 42,
      relationships: 88,
    },
    capture: {
      status: "ready",
      episodeCount: 12,
      activeCount: 0,
      latestEpisode: episode,
    },
    cue: {
      status: "attention",
      cueCount: 10,
      episodesWithoutCues: 2,
      coverage: 0.82,
      hitCount: 18,
      surfacedCount: 20,
      selectedCount: 9,
      usedCount: 7,
      nearMissCount: 1,
      avgPolicyScore: 0.73,
      projectionConversionRate: 0.8,
    },
    project: {
      status: "attention",
      projectedCount: 8,
      activeCount: 7,
      failedCount: 1,
      deadLetterCount: 0,
      failureRate: 0.09,
      stateCounts: {
        queued: 1,
        cued: 2,
        cueOnly: 1,
        scheduled: 3,
        projecting: 1,
        projected: 8,
        merged: 0,
        failed: 1,
        deadLetter: 0,
      },
    },
    recall: {
      status: "active",
      activeEntityCount: 1,
      topScore: 0.91,
      triggerCount: 0,
      intentions: {
        activeCount: 2,
        refreshContextCount: 1,
        afterConsolidationCount: 1,
        pinnedResultCount: 1,
        needsRefreshCount: 0,
        latestRefreshedAt: "2026-05-11T11:59:00Z",
      },
      topActivated: [
        {
          id: "ent_a",
          name: "Alice",
          entityType: "Person",
          summary: null,
          activation: 0.91,
          accessCount: 4,
        },
      ],
    },
    consolidate: {
      status: "attention",
      isRunning: false,
      schedulerActive: true,
      cycleCount: 1,
      pressure: {
        value: 0.42,
        threshold: 0.7,
        episodesSinceLast: 4,
        entitiesCreated: 2,
        lastCycleTime: null,
      },
      latestCycle: cycle,
    },
    recentEpisodes: [episode],
    ...overrides,
  };
}

const mockedApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
  mockedApi.getLifecycleSummary.mockResolvedValue(buildLifecycleSummary());
  mockedApi.getConsolidationHistory.mockResolvedValue({ cycles: [cycle] });
  mockedApi.getConsolidationStatus.mockResolvedValue({
    is_running: false,
    scheduler_active: true,
    pressure: {
      value: 0.42,
      threshold: 1,
      episodes_since_last: 4,
      entities_created: 2,
    },
  });
  act(() => {
    useEngramStore.setState({
      stats,
      isLoadingStats: false,
      lifecycleSummary: null,
      isLoadingLifecycleSummary: false,
      currentView: "lifecycle",
      lifecycleDrilldownStage: null,
      episodes: [episode],
      episodeCursor: null,
      hasMoreEpisodes: false,
      isLoadingEpisodes: false,
      knowledgeResults: [
        {
          resultType: "cue_episode",
          cue: {
            episodeId: "ep_1",
            cueText: "launch plan",
            supportingSpans: [],
            projectionState: "projected",
            routeReason: "salient",
            hitCount: 3,
            surfacedCount: 4,
            selectedCount: 2,
            usedCount: 1,
            nearMissCount: 0,
            policyScore: 0.74,
            lastFeedbackAt: null,
            lastProjectedAt: "2026-05-10T12:00:00Z",
          },
          episode: {
            id: "ep_1",
            source: "api",
            createdAt: "2026-05-10T12:00:00Z",
          },
          score: 0.86,
          scoreBreakdown: {
            semantic: 0.6,
            activation: 0.2,
            edgeProximity: 0.04,
            explorationBonus: 0.02,
          },
        },
      ],
      activationLeaderboard: [],
      cycles: [cycle],
      selectedCycleId: null,
      cycleDetail: null,
      isLoadingDetail: false,
      isRunning: false,
      schedulerActive: true,
      pressure: {
        value: 0.42,
        threshold: 1,
        episodes_since_last: 4,
        entities_created: 2,
      },
    });
  });
});

describe("LifecyclePanel", () => {
  it("renders the brain lifecycle stages from existing dashboard state", async () => {
    await act(async () => {
      render(<LifecyclePanel />);
      await Promise.resolve();
    });

    expect(screen.getByText("Brain Runtime")).toBeInTheDocument();
    expect(screen.getAllByText("Capture").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cue").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Project").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Recall").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Consolidate").length).toBeGreaterThan(0);
    expect(screen.getByText("82.0%")).toBeInTheDocument();
    expect(screen.getByText(/2 active intentions/)).toBeInTheDocument();
    expect(screen.getByText(/2\/3 phases/)).toBeInTheDocument();
    expect(screen.getByText(/calibration failed/)).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText(/Alice discussed the launch plan/)).toBeInTheDocument();
    expect(mockedApi.getLifecycleSummary).toHaveBeenCalled();
  });

  it("surfaces phase-level consolidation errors in fallback stage health", async () => {
    const phaseErrorCycle: ConsolidationCycleSummary = {
      ...cycle,
      status: "completed",
      error: null,
      phases: [
        {
          phase: "graph_embed",
          status: "error",
          items_processed: 1,
          items_affected: 0,
          duration_ms: 5,
          error: "optional vector index unavailable",
        },
      ],
    };
    mockedApi.getConsolidationHistory.mockResolvedValue({ cycles: [phaseErrorCycle] });
    act(() => {
      useEngramStore.setState({
        lifecycleSummary: null,
        isLoadingLifecycleSummary: true,
        cycles: [phaseErrorCycle],
        isRunning: false,
      });
    });

    await act(async () => {
      render(<LifecyclePanel />);
      await Promise.resolve();
    });

    expect(
      await screen.findByText(/graph_embed: optional vector index unavailable/),
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole("button", { name: "Open Consolidate drilldown" })).getByText(
        "attention",
      ),
    ).toBeInTheDocument();
    expect(mockedApi.getLifecycleSummary).not.toHaveBeenCalled();
  });

  it("opens existing drilldown views from lifecycle stage cards", async () => {
    const user = userEvent.setup();

    await act(async () => {
      render(<LifecyclePanel />);
      await Promise.resolve();
    });

    await user.click(screen.getByRole("button", { name: "Open Capture drilldown" }));
    expect(useEngramStore.getState().currentView).toBe("feed");
    expect(useEngramStore.getState().lifecycleDrilldownStage).toBe("capture");

    await user.click(screen.getByRole("button", { name: "Open Recall drilldown" }));
    expect(useEngramStore.getState().currentView).toBe("knowledge");
    expect(useEngramStore.getState().lifecycleDrilldownStage).toBe("recall");

    await user.click(screen.getByRole("button", { name: "Open Consolidate drilldown" }));
    expect(useEngramStore.getState().currentView).toBe("consolidation");
    expect(useEngramStore.getState().lifecycleDrilldownStage).toBe("consolidate");
    expect(useEngramStore.getState().selectedCycleId).toBe("cyc_1");
  });
});
