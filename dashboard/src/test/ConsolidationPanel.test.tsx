import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  CONSOLIDATION_PHASE_ORDER,
  type ConsolidationPhaseName,
} from "../constants/consolidation";
import type { ConsolidationPhaseResult } from "../store/types";

vi.mock("../api/client", () => ({
  api: {
    getConsolidationStatus: vi.fn(),
    getConsolidationHistory: vi.fn(),
    getConsolidationCycle: vi.fn(),
    triggerConsolidation: vi.fn(),
    // Other API mocks needed by the store
    getNeighborhood: vi.fn().mockResolvedValue({ centerId: "n1", nodes: [], edges: [], truncated: false, totalInNeighborhood: 0 }),
    searchEntities: vi.fn().mockResolvedValue([]),
    getLifecycleSummary: vi.fn().mockResolvedValue(null),
    getEvaluationReport: vi.fn().mockResolvedValue(null),
    recordRecallEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    recordSessionContinuityEvaluation: vi.fn().mockResolvedValue({ status: "stored" }),
    getStats: vi.fn().mockResolvedValue({ totalEntities: 0, totalRelationships: 0, totalEpisodes: 0, entityTypeCounts: {}, topActivated: [], topConnected: [], growthTimeline: [] }),
    getStorage: vi.fn().mockResolvedValue(null),
    getEpisodes: vi.fn().mockResolvedValue({ items: [], nextCursor: null }),
  },
}));

vi.stubGlobal("WebSocket", vi.fn(() => ({
  close: vi.fn(),
  send: vi.fn(),
  readyState: 1,
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
})));

import { ConsolidationPanel } from "../components/ConsolidationPanel";
import { useEngramStore } from "../store";
import { api } from "../api/client";

type PhaseResultOverride = Partial<Omit<ConsolidationPhaseResult, "phase" | "error">>;

const PHASE_RESULT_OVERRIDES: Record<ConsolidationPhaseName, PhaseResultOverride> = {
  triage: { items_processed: 20, items_affected: 6, duration_ms: 250 },
  merge: { items_processed: 5, items_affected: 1, duration_ms: 600 },
  calibrate: { items_processed: 8, items_affected: 2, duration_ms: 120 },
  infer: { items_processed: 3, items_affected: 2, duration_ms: 400 },
  evidence_adjudication: { status: "skipped", duration_ms: 5 },
  edge_adjudication: { status: "skipped", duration_ms: 5 },
  replay: { items_processed: 10, items_affected: 3, duration_ms: 800 },
  prune: { status: "skipped", duration_ms: 10 },
  compact: { items_processed: 8, items_affected: 8, duration_ms: 200 },
  mature: { items_processed: 4, items_affected: 1, duration_ms: 190 },
  semanticize: { items_processed: 6, items_affected: 1, duration_ms: 170 },
  schema: { status: "skipped", duration_ms: 12 },
  reindex: { items_processed: 2, items_affected: 2, duration_ms: 300 },
  graph_embed: { status: "skipped", duration_ms: 8 },
  microglia: { items_processed: 2, items_affected: 1, duration_ms: 90 },
  immunity: { items_processed: 3, items_affected: 1, duration_ms: 75 },
  dream: { status: "skipped", duration_ms: 5 },
};

function buildPhaseResults(): ConsolidationPhaseResult[] {
  return CONSOLIDATION_PHASE_ORDER.map((phase) => {
    const override = PHASE_RESULT_OVERRIDES[phase];
    return {
      phase,
      status: override.status ?? "success",
      items_processed: override.items_processed ?? 0,
      items_affected: override.items_affected ?? 0,
      duration_ms: override.duration_ms ?? 0,
      error: null,
    };
  });
}

function configureApiMocks() {
  const phases = buildPhaseResults();
  const startedAt = Date.now() / 1000 - 300;
  const completedAt = Date.now() / 1000 - 295;
  const cycleSummary = {
    id: "cyc_abc123",
    status: "failed" as const,
    error: "calibration failed",
    phase_issue: null,
    dry_run: false,
    trigger: "manual",
    started_at: startedAt,
    completed_at: completedAt,
    total_duration_ms: 5200,
    phases,
  };

  vi.mocked(api.getConsolidationStatus).mockResolvedValue({
    is_running: false,
    scheduler_active: true,
    pressure: { value: 0.45, threshold: 1.0, episodes_since_last: 12, entities_created: 5 },
  });
  vi.mocked(api.getConsolidationHistory).mockResolvedValue({ cycles: [cycleSummary] });
  vi.mocked(api.getConsolidationCycle).mockResolvedValue({
    ...cycleSummary,
    error: null,
    merges: [
      {
        id: "mrg_1",
        keep_name: "Alice",
        remove_name: "alice",
        similarity: 0.92,
        decision_confidence: 0.99,
        decision_source: "identifier_policy",
        decision_reason: "identifier_exact_match",
        relationships_transferred: 3,
      },
    ],
    identifier_reviews: [
      {
        id: "idr_1",
        entity_a_name: "1712061",
        entity_b_name: "1712018",
        entity_a_type: "Identifier",
        entity_b_type: "Identifier",
        raw_similarity: 0.86,
        adjusted_similarity: 0.89,
        decision_source: "fuzzy_threshold",
        decision_reason: "identifier_mismatch",
        canonical_identifier_a: "1712061",
        canonical_identifier_b: "1712018",
        review_status: "quarantined",
        metadata: {},
      },
    ],
    inferred_edges: [],
    prunes: [],
    dreams: [],
    replays: [],
    reindexes: [],
  });
  vi.mocked(api.triggerConsolidation).mockResolvedValue({
    status: "triggered",
    group_id: "default",
    dry_run: true,
  });
}

function resetStore() {
  useEngramStore.setState({
    cycles: [],
    isLoadingCycles: false,
    selectedCycleId: null,
    cycleDetail: null,
    isLoadingDetail: false,
    isRunning: false,
    schedulerActive: false,
    pressure: null,
    currentView: "consolidation",
    readyState: "disconnected",
    lastSeq: 0,
    reconnectAttempt: 0,
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  configureApiMocks();
  act(() => {
    resetStore();
  });
});

describe("ConsolidationPanel", () => {
  it("renders loading state", async () => {
    act(() => {
      useEngramStore.setState({ isLoadingCycles: true, cycles: [] });
    });
    const { container } = render(<ConsolidationPanel />);
    expect(container.querySelector(".skeleton")).toBeInTheDocument();
  });

  it("renders cycle list after load", async () => {
    render(<ConsolidationPanel />);
    // loadCycles is called on mount, which resolves with our mock
    expect(await screen.findByText("manual")).toBeInTheDocument();
    expect(screen.getByText("calibration failed")).toBeInTheDocument();
  });

  it("renders phase issue in cycle list when cycle error is empty", async () => {
    vi.mocked(api.getConsolidationHistory).mockResolvedValueOnce({
      cycles: [
        {
          id: "cyc_phase_issue",
          status: "completed",
          error: null,
          phase_issue: "graph_embed: optional vector index unavailable",
          dry_run: false,
          trigger: "manual",
          started_at: Date.now() / 1000 - 120,
          completed_at: Date.now() / 1000 - 115,
          total_duration_ms: 5000,
          phases: [],
        },
      ],
    });

    render(<ConsolidationPanel />);

    expect(
      await screen.findByText("graph_embed: optional vector index unavailable"),
    ).toBeInTheDocument();
  });

  it("selecting a cycle loads detail", async () => {
    render(<ConsolidationPanel />);
    // Wait for cycles to load
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(api.getConsolidationCycle).toHaveBeenCalledWith("cyc_abc123");
  });

  it("phase timeline shows all 17 phases", async () => {
    render(<ConsolidationPanel />);
    // Click cycle to load detail
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(CONSOLIDATION_PHASE_ORDER).toHaveLength(17);
    expect(await screen.findByText("triage")).toBeInTheDocument();
    CONSOLIDATION_PHASE_ORDER.forEach((phase) => {
      expect(screen.getByText(phase)).toBeInTheDocument();
    });
  });

  it("phase timeline shows phase error text", async () => {
    const phases = buildPhaseResults().map((phase) =>
      phase.phase === "graph_embed"
        ? {
            ...phase,
            status: "error" as const,
            error: "optional vector index unavailable",
          }
        : phase,
    );
    vi.mocked(api.getConsolidationCycle).mockResolvedValueOnce({
      id: "cyc_abc123",
      status: "completed",
      error: null,
      phase_issue: "graph_embed: optional vector index unavailable",
      dry_run: false,
      trigger: "manual",
      started_at: Date.now() / 1000 - 300,
      completed_at: Date.now() / 1000 - 295,
      total_duration_ms: 5200,
      phases,
      merges: [],
      identifier_reviews: [],
      inferred_edges: [],
      prunes: [],
      dreams: [],
      replays: [],
      reindexes: [],
    });

    render(<ConsolidationPanel />);
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(await screen.findByText("warning")).toBeInTheDocument();
    expect(await screen.findByText("optional vector index unavailable")).toBeInTheDocument();
  });

  it("merge details distinguish decision confidence from name similarity", async () => {
    render(<ConsolidationPanel />);
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(await screen.findByText("decision 99%")).toBeInTheDocument();
    expect(screen.getByText("name 92%")).toBeInTheDocument();
    expect(screen.getByText("identifier exact")).toBeInTheDocument();
  });

  it("renders quarantined identifier review records", async () => {
    render(<ConsolidationPanel />);
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(await screen.findByText("Identifier Reviews")).toBeInTheDocument();
    expect(screen.getByText("identifier mismatch")).toBeInTheDocument();
    expect(screen.getByText("raw 86%")).toBeInTheDocument();
    expect(screen.getByText("quarantined")).toBeInTheDocument();
  });

  it("trigger button calls API", async () => {
    render(<ConsolidationPanel />);
    const user = userEvent.setup();
    const triggerBtn = await screen.findByText("Trigger Cycle");
    await user.click(triggerBtn);

    expect(api.triggerConsolidation).toHaveBeenCalledWith(true); // dry_run=true default
  });

  it("pressure gauge renders when available", async () => {
    render(<ConsolidationPanel />);
    // loadStatus is called on mount, which resolves with pressure data
    expect(await screen.findByText("Pressure")).toBeInTheDocument();
    expect(screen.getByText(/0\.45/)).toBeInTheDocument();
  });
});
