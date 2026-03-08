import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("../api/client", () => ({
  api: {
    getConsolidationStatus: vi.fn().mockResolvedValue({
      is_running: false,
      scheduler_active: true,
      pressure: { value: 0.45, threshold: 1.0, episodes_since_last: 12, entities_created: 5 },
    }),
    getConsolidationHistory: vi.fn().mockResolvedValue({
      cycles: [
        {
          id: "cyc_abc123",
          status: "completed",
          dry_run: false,
          trigger: "manual",
          started_at: Date.now() / 1000 - 300,
          completed_at: Date.now() / 1000 - 295,
          total_duration_ms: 5200,
          phases: [
            { phase: "replay", status: "success", items_processed: 10, items_affected: 3, duration_ms: 800, error: null },
            { phase: "merge", status: "success", items_processed: 5, items_affected: 1, duration_ms: 600, error: null },
            { phase: "infer", status: "success", items_processed: 3, items_affected: 2, duration_ms: 400, error: null },
            { phase: "prune", status: "skipped", items_processed: 0, items_affected: 0, duration_ms: 10, error: null },
            { phase: "compact", status: "success", items_processed: 8, items_affected: 8, duration_ms: 200, error: null },
            { phase: "reindex", status: "success", items_processed: 2, items_affected: 2, duration_ms: 300, error: null },
            { phase: "dream", status: "skipped", items_processed: 0, items_affected: 0, duration_ms: 5, error: null },
          ],
        },
      ],
    }),
    getConsolidationCycle: vi.fn().mockResolvedValue({
      id: "cyc_abc123",
      status: "completed",
      dry_run: false,
      trigger: "manual",
      started_at: Date.now() / 1000 - 300,
      completed_at: Date.now() / 1000 - 295,
      total_duration_ms: 5200,
      error: null,
      phases: [
        { phase: "replay", status: "success", items_processed: 10, items_affected: 3, duration_ms: 800, error: null },
        { phase: "merge", status: "success", items_processed: 5, items_affected: 1, duration_ms: 600, error: null },
        { phase: "infer", status: "success", items_processed: 3, items_affected: 2, duration_ms: 400, error: null },
        { phase: "prune", status: "skipped", items_processed: 0, items_affected: 0, duration_ms: 10, error: null },
        { phase: "compact", status: "success", items_processed: 8, items_affected: 8, duration_ms: 200, error: null },
        { phase: "reindex", status: "success", items_processed: 2, items_affected: 2, duration_ms: 300, error: null },
        { phase: "dream", status: "skipped", items_processed: 0, items_affected: 0, duration_ms: 5, error: null },
      ],
      merges: [{
        id: "mrg_1",
        keep_name: "Alice",
        remove_name: "alice",
        similarity: 0.92,
        decision_confidence: 0.99,
        decision_source: "identifier_policy",
        decision_reason: "identifier_exact_match",
        relationships_transferred: 3,
      }],
      identifier_reviews: [{
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
      }],
      inferred_edges: [],
      prunes: [],
      dreams: [],
      replays: [],
      reindexes: [],
    }),
    triggerConsolidation: vi.fn().mockResolvedValue({ status: "triggered", cycle_id: "cyc_new" }),
    // Other API mocks needed by the store
    getNeighborhood: vi.fn().mockResolvedValue({ centerId: "n1", nodes: [], edges: [], truncated: false, totalInNeighborhood: 0 }),
    searchEntities: vi.fn().mockResolvedValue([]),
    getStats: vi.fn().mockResolvedValue({ totalEntities: 0, totalRelationships: 0, totalEpisodes: 0, entityTypeCounts: {}, topActivated: [], topConnected: [], growthTimeline: [] }),
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
  });

  it("selecting a cycle loads detail", async () => {
    render(<ConsolidationPanel />);
    // Wait for cycles to load
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    expect(api.getConsolidationCycle).toHaveBeenCalledWith("cyc_abc123");
  });

  it("phase timeline shows all 7 phases", async () => {
    render(<ConsolidationPanel />);
    // Click cycle to load detail
    const cycleButton = await screen.findByText("manual");
    const user = userEvent.setup();
    await user.click(cycleButton);

    // Wait for detail to load
    expect(await screen.findByText("replay")).toBeInTheDocument();
    expect(screen.getByText("merge")).toBeInTheDocument();
    expect(screen.getByText("infer")).toBeInTheDocument();
    expect(screen.getByText("prune")).toBeInTheDocument();
    expect(screen.getByText("compact")).toBeInTheDocument();
    expect(screen.getByText("reindex")).toBeInTheDocument();
    expect(screen.getByText("dream")).toBeInTheDocument();
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
