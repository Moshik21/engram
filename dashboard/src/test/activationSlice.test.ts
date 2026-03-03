import { describe, it, expect, vi, beforeEach } from "vitest";
import { createStore } from "zustand";
import { immer } from "zustand/middleware/immer";
import type { EngramStore, ActivationItem } from "../store/types";
import { createGraphSlice } from "../store/graphSlice";
import { createSelectionSlice } from "../store/selectionSlice";
import { createPreferencesSlice } from "../store/preferencesSlice";
import { createTimeSlice } from "../store/timeSlice";
import { createEpisodeSlice } from "../store/episodeSlice";
import { createStatsSlice } from "../store/statsSlice";
import { createWsSlice } from "../store/wsSlice";
import { createActivationSlice } from "../store/activationSlice";
import { createConsolidationSlice } from "../store/consolidationSlice";

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
    })),
  );
}

function makeItem(overrides: Partial<ActivationItem> = {}): ActivationItem {
  return {
    entityId: "e1",
    name: "Alice",
    entityType: "Person",
    currentActivation: 0.75,
    accessCount: 5,
    lastAccessedAt: "2024-01-01T00:00:00Z",
    decayRate: 0.5,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("activationSlice", () => {
  it("setActivationLeaderboard stores data", () => {
    const store = createTestStore();
    const items = [makeItem(), makeItem({ entityId: "e2", name: "Bob" })];

    store.getState().setActivationLeaderboard(items);

    expect(store.getState().activationLeaderboard).toHaveLength(2);
    expect(store.getState().activationLeaderboard[0].name).toBe("Alice");
    expect(store.getState().activationLeaderboard[1].name).toBe("Bob");
  });

  it("selectActivationEntity sets selectedEntityId", () => {
    const store = createTestStore();

    expect(store.getState().selectedActivationEntity).toBeNull();

    store.getState().selectActivationEntity("e1");
    expect(store.getState().selectedActivationEntity).toBe("e1");

    store.getState().selectActivationEntity(null);
    expect(store.getState().selectedActivationEntity).toBeNull();
  });

  it("loadDecayCurve happy path sets curve data", async () => {
    const store = createTestStore();

    mockedApi.getActivationCurve.mockResolvedValueOnce({
      entityId: "e1",
      entityName: "Alice",
      curve: [
        { timestamp: "2024-01-01T00:00:00Z", activation: 0.9 },
        { timestamp: "2024-01-01T01:00:00Z", activation: 0.7 },
      ],
      accessEvents: ["2024-01-01T00:00:00Z"],
      formula: "B_i = ln(Σ t_j^{-0.5})",
      hours: 24,
      points: 50,
    });

    await store.getState().loadDecayCurve("e1");

    expect(store.getState().decayCurve).toHaveLength(2);
    expect(store.getState().decayFormula).toBe("B_i = ln(Σ t_j^{-0.5})");
    expect(store.getState().accessEvents).toEqual(["2024-01-01T00:00:00Z"]);
    expect(store.getState().isLoadingCurve).toBe(false);
  });

  it("loadDecayCurve error does not crash", async () => {
    const store = createTestStore();

    mockedApi.getActivationCurve.mockRejectedValueOnce(new Error("Network error"));

    await store.getState().loadDecayCurve("e1");

    expect(store.getState().isLoadingCurve).toBe(false);
    // Curve data should remain at defaults
    expect(store.getState().decayCurve).toEqual([]);
  });

  it("setIsActivationSubscribed toggles flag", () => {
    const store = createTestStore();

    expect(store.getState().isActivationSubscribed).toBe(false);

    store.getState().setIsActivationSubscribed(true);
    expect(store.getState().isActivationSubscribed).toBe(true);

    store.getState().setIsActivationSubscribed(false);
    expect(store.getState().isActivationSubscribed).toBe(false);
  });

  it("addActivationPulse appends to pulse array", () => {
    const store = createTestStore();

    expect(store.getState().activationPulses).toHaveLength(0);

    store.getState().addActivationPulse({
      entityId: "e1",
      name: "Alice",
      entityType: "Person",
      activation: 0.8,
      accessedVia: "recall",
    });

    expect(store.getState().activationPulses).toHaveLength(1);
    expect(store.getState().activationPulses[0].entityId).toBe("e1");
    expect(store.getState().activationPulses[0].accessedVia).toBe("recall");
    expect(store.getState().activationPulses[0].timestamp).toBeGreaterThan(0);
  });

  it("addActivationPulse caps at 50 entries", () => {
    const store = createTestStore();

    for (let i = 0; i < 60; i++) {
      store.getState().addActivationPulse({
        entityId: `e${i}`,
        name: `Entity${i}`,
        entityType: "Other",
        activation: 0.5,
        accessedVia: "ingest",
      });
    }

    expect(store.getState().activationPulses).toHaveLength(50);
    // Should keep the last 50
    expect(store.getState().activationPulses[0].entityId).toBe("e10");
  });

  it("clearExpiredPulses removes old pulses", () => {
    const store = createTestStore();
    const now = Date.now();

    // Manually set pulses with controlled timestamps
    store.getState().addActivationPulse({
      entityId: "e1",
      name: "Old",
      entityType: "Person",
      activation: 0.5,
      accessedVia: "ingest",
    });

    // The pulse was just added so it should NOT be cleared with default maxAge
    store.getState().clearExpiredPulses(2000);
    expect(store.getState().activationPulses).toHaveLength(1);

    // Clear with maxAge=0 should remove everything
    store.getState().clearExpiredPulses(0);
    expect(store.getState().activationPulses).toHaveLength(0);
  });
});
