import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { ConsolidationPanel } from "../components/ConsolidationPanel";
import { EvaluationPanel } from "../components/EvaluationPanel";
import { LifecyclePanel } from "../components/LifecyclePanel";
import { MemoryFeed } from "../components/MemoryFeed";
import { useEngramStore } from "../store";

const shouldRunNativeSmoke = import.meta.env.VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE === "1";
const describeNativeSmoke = shouldRunNativeSmoke ? describe : describe.skip;

const nativeLifecyclePayload = {
  groupId: "native_brain",
  generatedAt: "2026-05-12T20:56:46.473012Z",
  loop: ["capture", "cue", "project", "recall", "consolidate"],
  totals: {
    episodes: 3,
    cues: 3,
    projected: 3,
    cycles: 1,
    entities: 2,
    relationships: 0,
  },
  capture: {
    status: "ready",
    episodeCount: 3,
    activeCount: 0,
    latestEpisode: {
      episodeId: "ep_native_3",
      content:
        "Konner prefers concise engineering updates and uses Engram to remember architecture decisions.",
      source: "projected-consolidated-smoke",
      status: "completed",
      projectionState: "projected",
      lastProjectionReason: "projected",
      lastProjectedAt: "2026-05-12T20:56:35.626440Z",
      conversationDate: null,
      createdAt: "2026-05-12T20:56:35.486612Z",
      updatedAt: "2026-05-12T20:56:35.626477Z",
      entities: [],
      factsCount: 0,
      processingDurationMs: 78,
      error: null,
      retryCount: 0,
      cue: {
        cueText: "mentions: Konner, Engram",
        projectionState: "projected",
        routeReason: null,
        hitCount: 0,
        surfacedCount: 0,
        selectedCount: 0,
        usedCount: 0,
        nearMissCount: 0,
        policyScore: 0,
        projectionAttempts: 0,
        lastHitAt: null,
        lastFeedbackAt: null,
        lastProjectedAt: null,
      },
    },
  },
  cue: {
    status: "ready",
    cueCount: 3,
    episodesWithoutCues: 0,
    coverage: 1,
    hitCount: 0,
    surfacedCount: 0,
    selectedCount: 0,
    usedCount: 0,
    nearMissCount: 0,
    avgPolicyScore: 0,
    projectionConversionRate: 1,
  },
  project: {
    status: "ready",
    projectedCount: 3,
    activeCount: 0,
    failedCount: 0,
    deadLetterCount: 0,
    failureRate: 0,
    stateCounts: {
      queued: 0,
      cued: 0,
      cueOnly: 0,
      scheduled: 0,
      projecting: 0,
      projected: 3,
      merged: 0,
      failed: 0,
      deadLetter: 0,
    },
  },
  recall: {
    status: "ready",
    activeEntityCount: 0,
    topScore: 0,
    triggerCount: 0,
    intentions: {
      activeCount: 0,
      refreshContextCount: 0,
      afterConsolidationCount: 0,
      pinnedResultCount: 0,
      needsRefreshCount: 0,
      latestRefreshedAt: null,
    },
    topActivated: [],
  },
  consolidate: {
    status: "attention",
    isRunning: false,
    schedulerActive: false,
    cycleCount: 1,
    pressure: null,
    latestCycle: {
      id: "cyc_native_1",
      status: "completed",
      dry_run: false,
      trigger: "projected_consolidated_smoke",
      started_at: 1778619395.505337,
      completed_at: 1778619396.5332549,
      total_duration_ms: 1027.9,
      error: null,
      phase_issue: "graph_embed: optional vector index unavailable",
      phases: [
        {
          phase: "triage",
          status: "success",
          items_processed: 3,
          items_affected: 3,
          duration_ms: 917.1,
          error: null,
        },
        {
          phase: "graph_embed",
          status: "error",
          items_processed: 0,
          items_affected: 0,
          duration_ms: 3.4,
          error: "optional vector index unavailable",
        },
      ],
    },
  },
  recentEpisodes: [],
};

const nativeConsolidationCycle = {
  id: "cyc_native_1",
  status: "completed",
  dry_run: false,
  trigger: "projected_consolidated_smoke",
  started_at: 1778619395.505337,
  completed_at: 1778619396.5332549,
  total_duration_ms: 1027.9,
  error: null,
  phase_issue: "graph_embed: optional vector index unavailable",
  phases: [
    {
      phase: "triage",
      status: "success",
      items_processed: 3,
      items_affected: 3,
      duration_ms: 917.1,
      error: null,
    },
    {
      phase: "graph_embed",
      status: "error",
      items_processed: 0,
      items_affected: 0,
      duration_ms: 3.4,
      error: "optional vector index unavailable",
    },
  ],
};

const nativeConsolidationCycleDetail = {
  ...nativeConsolidationCycle,
  merges: [],
  identifier_reviews: [],
  inferred_edges: [],
  prunes: [],
  dreams: [],
  replays: [],
  reindexes: [],
};

const nativeConsolidationStatusPayload = {
  is_running: false,
  scheduler_active: false,
  latest_cycle: nativeConsolidationCycle,
};

const nativeConsolidationHistoryPayload = {
  cycles: [nativeConsolidationCycle],
};

const nativeEpisodesPayload = {
  items: [
    {
      episodeId: "ep_native_queued",
      content: "Native queued cue payload keeps Capture and Cue visible in the dashboard.",
      source: "native-rest-observe",
      status: "queued",
      projectionState: "cued",
      lastProjectionReason: "triage_selected",
      lastProjectedAt: null,
      conversationDate: "2026-05-13T13:00:00Z",
      createdAt: "2026-05-13T13:00:00Z",
      updatedAt: "2026-05-13T13:00:01Z",
      entities: [],
      factsCount: 0,
      processingDurationMs: null,
      error: null,
      retryCount: 0,
      cue: {
        cueText: "mentions: Native, Cue",
        projectionState: "cued",
        routeReason: "triage_selected",
        hitCount: 1,
        surfacedCount: 1,
        selectedCount: 1,
        usedCount: 0,
        nearMissCount: 0,
        policyScore: 0.73,
        projectionAttempts: 1,
        lastHitAt: "2026-05-13T13:00:02Z",
        lastFeedbackAt: null,
        lastProjectedAt: null,
      },
    },
  ],
  nextCursor: null,
  total: 1,
};

const nativeEvaluationPayload = {
  group_id: "native_brain",
  generated_at: "2026-05-12T20:56:36.609187Z",
  loop: ["capture", "cue", "project", "recall", "consolidate"],
  totals: { episodes: 3, entities: 2, relationships: 0, active_entities: 0 },
  capture: { status: "ready", episode_count: 3, active_count: 0 },
  cue: {
    status: "ready",
    cue_count: 3,
    episodes_without_cues: 0,
    coverage: 1,
    hit_count: 0,
    hit_episode_count: 0,
    hit_episode_rate: 0,
    surfaced_count: 0,
    selected_count: 0,
    used_count: 0,
    near_miss_count: 0,
    selected_rate: 0,
    used_rate: 0,
    near_miss_rate: 0,
    avg_policy_score: 0,
    projection_conversion_rate: 1,
  },
  project: {
    status: "ready",
    state_counts: { projected: 3, failed: 0, dead_letter: 0, cue_only: 0 },
    projected_count: 3,
    active_count: 0,
    failed_count: 0,
    dead_letter_count: 0,
    attempted_episode_count: 3,
    total_attempts: 3,
    failure_rate: 0,
    yield: {
      linked_entity_count: 3,
      relationship_count: 0,
      avg_linked_entities_per_projected_episode: 1,
      avg_relationships_per_projected_episode: 0,
    },
  },
  recall: {
    status: "active",
    total_analyses: 1,
    trigger_count: 1,
    latency: {
      analyzer_ms: { avg_ms: 1.2666, p95_ms: 1.2666 },
      probe_ms: { avg_ms: 0, p95_ms: 0 },
    },
    control: {
      used_count: 0,
      dismissed_count: 0,
      surfaced_count: 1,
      selected_count: 0,
      confirmed_count: 0,
      corrected_count: 0,
      graph_override_count: 0,
      adaptive_thresholds_enabled: false,
      thresholds: { linguistic: 0.3, borderline: 0.15, resonance: 0.45 },
    },
    family_contributions: { keyword: 1 },
    evaluation: {
      status: "measured",
      sample_count: 1,
      memory_need_precision: 1,
      useful_packet_rate: 0.6667,
      false_recall_rate: 0,
      surfaced_count: 3,
      used_count: 2,
      surfaced_to_used_ratio: 1.5,
    },
    continuity: {
      status: "measured",
      sample_count: 1,
      session_continuity_lift: 0.6,
      open_loop_recovery_rate: 1,
      temporal_correctness: 1,
    },
  },
  consolidate: {
    status: "attention",
    cycle_count: 1,
    latest_status: "completed",
    latest_cycle: {
      id: "cyc_native_1",
      status: "completed",
      error: null,
      phase_issue: "graph_embed: optional vector index unavailable",
    },
    phase_status_counts: { success: 1, error: 1 },
    phase_totals: {
      triage: { runs: 1, items_processed: 3, items_affected: 3 },
      graph_embed: { runs: 1, items_processed: 0, items_affected: 0 },
    },
    calibration: {
      status: "measured",
      snapshot_count: 1,
      phase_totals: { triage: { snapshots: 1, total_traces: 3, labeled_examples: 3 } },
    },
    items_processed: 3,
    items_affected: 3,
    error_count: 1,
  },
  coverage_gaps: [],
};

const nativeRecallPayload = {
  query: "Engram brain loop",
  items: [
    {
      resultType: "entity",
      entity: {
        id: "ent_engram",
        name: "Engram",
        entityType: "Technology",
        summary: "Memory runtime for AI agents",
      },
      score: 0.93,
      scoreBreakdown: {
        semantic: 0.7,
        activation: 0.1,
        edgeProximity: 0.08,
        explorationBonus: 0.05,
      },
      relationships: [],
    },
  ],
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function resetDashboardState() {
  act(() => {
    useEngramStore.setState({
      currentView: "lifecycle",
      lifecycleDrilldownStage: null,
      lifecycleSummary: null,
      isLoadingLifecycleSummary: false,
      stats: null,
      isLoadingStats: false,
      episodes: [],
      episodeCursor: null,
      hasMoreEpisodes: false,
      isLoadingEpisodes: false,
      knowledgeResults: [],
      activationLeaderboard: [],
      cycles: [],
      selectedCycleId: null,
      cycleDetail: null,
      isLoadingDetail: false,
      isRunning: false,
      schedulerActive: false,
      pressure: null,
    });
  });
}

function mockNativeDashboardApi() {
  const fetchMock = vi.fn(async (input: unknown) => {
    const url = String(input);
    if (url.includes("/api/lifecycle/summary")) {
      return { ok: true, json: async () => nativeLifecyclePayload };
    }
    if (url.includes("/api/evaluation/brain-loop/report")) {
      return { ok: true, json: async () => nativeEvaluationPayload };
    }
    if (url.includes("/api/consolidation/status")) {
      return { ok: true, json: async () => nativeConsolidationStatusPayload };
    }
    if (url.includes("/api/consolidation/history")) {
      return { ok: true, json: async () => nativeConsolidationHistoryPayload };
    }
    if (url.includes("/api/consolidation/cycle/")) {
      return { ok: true, json: async () => nativeConsolidationCycleDetail };
    }
    if (url.includes("/api/episodes")) {
      return { ok: true, json: async () => nativeEpisodesPayload };
    }
    if (url.includes("/api/knowledge/recall")) {
      return { ok: true, json: async () => nativeRecallPayload };
    }
    throw new Error(`Unexpected native dashboard fixture request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("native PyO3 dashboard fixture smoke", () => {
  it("renders the brain lifecycle from native-shaped API payloads without a REST bind", async () => {
    const fetchMock = mockNativeDashboardApi();
    resetDashboardState();

    const [
      lifecycle,
      report,
      recall,
      consolidationStatus,
      consolidationHistory,
      episodes,
    ] = await Promise.all([
      api.getLifecycleSummary(),
      api.getEvaluationReport(),
      api.recall({ q: "Engram brain loop", limit: 5 }),
      api.getConsolidationStatus(),
      api.getConsolidationHistory(),
      api.getEpisodes({ limit: 20 }),
    ]);

    expect(lifecycle.groupId).toBe("native_brain");
    expect(lifecycle.totals).toMatchObject({ episodes: 3, cues: 3, projected: 3, cycles: 1 });
    expect(report.groupId).toBe("native_brain");
    expect(report.coverageGaps).toEqual([]);
    expect(report.project.projectedCount).toBe(3);
    expect(report.project.yield.linkedEntityCount).toBe(3);
    expect(report.recall.evaluation.status).toBe("measured");
    expect(report.recall.continuity.status).toBe("measured");
    expect(report.recall.triggerCount).toBe(1);
    expect(report.recall.latency.analyzerMs.p95Ms).toBe(1.2666);
    expect(report.recall.control.surfacedCount).toBe(1);
    expect(report.recall.control.thresholds.resonance).toBe(0.45);
    expect(recall.items.length).toBe(1);
    expect(consolidationStatus.latest_cycle?.id).toBe("cyc_native_1");
    expect(consolidationStatus.latest_cycle?.phases[0].phase).toBe("triage");
    expect(consolidationHistory.cycles[0].trigger).toBe("projected_consolidated_smoke");
    expect(episodes.items[0].projectionState).toBe("cued");
    expect(episodes.items[0].cue?.projectionState).toBe("cued");
    expect(episodes.items[0].cue?.policyScore).toBe(0.73);

    act(() => {
      useEngramStore.setState({
        evaluationReport: report,
        lifecycleSummary: lifecycle,
        knowledgeResults: recall.items,
      });
    });

    render(<LifecyclePanel />);

    expect(await screen.findByText("Brain Runtime")).toBeInTheDocument();
    await waitFor(() => {
      expect(useEngramStore.getState().lifecycleSummary?.groupId).toBe("native_brain");
    });

    expect(screen.getByText(/3 episodes/)).toBeInTheDocument();
    expect(screen.getByText(/3 cues/)).toBeInTheDocument();
    expect(screen.getByText(/3 projected/)).toBeInTheDocument();
    expect(screen.getByText(/1 cycles/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/lifecycle/summary"),
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/evaluation/brain-loop/report"),
      expect.any(Object),
    );

    render(<EvaluationPanel />);

    expect(await screen.findByText("Runtime quality signals")).toBeInTheDocument();
    expect(screen.getByText("Recall Gate")).toBeInTheDocument();
    expect(screen.getByText("analysis p95")).toBeInTheDocument();
    expect(screen.getByText("runtime used")).toBeInTheDocument();
    expect(screen.getByText("resonance")).toBeInTheDocument();
    expect(screen.getByText("0.45")).toBeInTheDocument();

    render(<ConsolidationPanel />);

    expect(await screen.findByText("Consolidation Status")).toBeInTheDocument();
    await waitFor(() => {
      expect(useEngramStore.getState().cycles[0]?.id).toBe("cyc_native_1");
    });
    expect(screen.getByText("IDLE")).toBeInTheDocument();
    expect(screen.getByText("Scheduler OFF")).toBeInTheDocument();
    expect(screen.getByText("projected_consolidated_smoke")).toBeInTheDocument();
    expect(
      screen.getAllByText("graph_embed: optional vector index unavailable").length,
    ).toBeGreaterThanOrEqual(1);

    await userEvent.click(screen.getByText("projected_consolidated_smoke"));
    expect(await screen.findByText("warning")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByText("optional vector index unavailable").length).toBeGreaterThanOrEqual(1);
    });

    render(<MemoryFeed />);

    expect(await screen.findByText(/Native queued cue payload/)).toBeInTheDocument();
    await waitFor(() => {
      expect(useEngramStore.getState().episodes[0]?.cue?.policyScore).toBe(0.73);
    });
    expect(screen.getByText("native-rest-observe")).toBeInTheDocument();
  });
});

describeNativeSmoke("native PyO3 dashboard smoke", () => {
  it("renders the brain lifecycle from a populated native Helix backend", async () => {
    if (!import.meta.env.VITE_API_URL) {
      throw new Error("Set VITE_API_URL to the running Engram REST server");
    }

    resetDashboardState();

    const [
      lifecycle,
      report,
      recall,
      consolidationStatus,
      consolidationHistory,
      episodes,
    ] = await Promise.all([
      api.getLifecycleSummary(),
      api.getEvaluationReport(),
      api.recall({ q: "Engram brain loop", limit: 5 }),
      api.getConsolidationStatus(),
      api.getConsolidationHistory(),
      api.getEpisodes({ limit: 20 }),
    ]);

    expect(lifecycle.groupId).toBe("native_brain");
    expect(lifecycle.loop).toEqual(["capture", "cue", "project", "recall", "consolidate"]);
    expect(lifecycle.totals.episodes).toBe(3);
    expect(lifecycle.totals.cues).toBe(3);
    expect(lifecycle.totals.projected).toBe(3);
    expect(lifecycle.totals.cycles).toBeGreaterThanOrEqual(1);
    expect(lifecycle.recentEpisodes.length).toBeGreaterThan(0);

    expect(report.groupId).toBe("native_brain");
    expect(report.coverageGaps).toEqual([]);
    expect(report.project.projectedCount).toBe(3);
    expect(report.project.yield.linkedEntityCount).toBeGreaterThan(0);
    expect(report.recall.evaluation.status).toBe("measured");
    expect(report.recall.continuity.status).toBe("measured");

    expect(recall.items.length).toBeGreaterThan(0);
    expect(consolidationStatus.latest_cycle?.id).toBeTruthy();
    expect(consolidationStatus.latest_cycle?.phases.length).toBeGreaterThan(0);
    expect(consolidationHistory.cycles.length).toBeGreaterThan(0);
    expect(episodes.items.length).toBeGreaterThan(0);
    expect(episodes.items[0].cue).toBeTruthy();

    act(() => {
      useEngramStore.setState({ knowledgeResults: recall.items });
    });

    render(<LifecyclePanel />);

    expect(await screen.findByText("Brain Runtime")).toBeInTheDocument();
    await waitFor(() => {
      expect(useEngramStore.getState().lifecycleSummary?.groupId).toBe("native_brain");
    });

    expect(screen.getByText(/3 episodes/)).toBeInTheDocument();
    expect(screen.getByText(/3 cues/)).toBeInTheDocument();
    expect(screen.getByText(/3 projected/)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${lifecycle.totals.cycles} cycles`))).toBeInTheDocument();
    expect(screen.getAllByText("Capture").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cue").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Project").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Recall").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Consolidate").length).toBeGreaterThan(0);
  });
});
