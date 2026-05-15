import type { StateCreator } from "zustand";
import type { EngramStore, PreferencesSlice } from "./types";

export const createPreferencesSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  PreferencesSlice
> = (set) => ({
  currentView: "lifecycle",
  lifecycleDrilldownStage: null,
  renderMode: "3d",
  showActivationHeatmap: true,
  showEdgeLabels: false,
  showFpsOverlay: false,
  darkMode: true,
  graphMaxNodes: 1500,
  lastAtlasVisitAt: null,
  lastAtlasSnapshotId: null,
  dashboardMode: "observatory" as const,

  setCurrentView: (view) =>
    set((s) => {
      s.currentView = view;
      s.lifecycleDrilldownStage = null;
    }),
  setLifecycleDrilldownStage: (stage) =>
    set((s) => {
      s.lifecycleDrilldownStage = stage;
    }),
  setRenderMode: (mode) =>
    set((s) => {
      s.renderMode = mode;
    }),
  toggleActivationHeatmap: () =>
    set((s) => {
      s.showActivationHeatmap = !s.showActivationHeatmap;
    }),
  toggleEdgeLabels: () =>
    set((s) => {
      s.showEdgeLabels = !s.showEdgeLabels;
    }),
  toggleFpsOverlay: () =>
    set((s) => {
      s.showFpsOverlay = !s.showFpsOverlay;
    }),
  toggleDarkMode: () =>
    set((s) => {
      s.darkMode = !s.darkMode;
    }),
  setGraphMaxNodes: (n) =>
    set((s) => {
      s.graphMaxNodes = n;
    }),
  recordAtlasVisit: ({ generatedAt, snapshotId }) =>
    set((s) => {
      if (!generatedAt) return;
      s.lastAtlasVisitAt = generatedAt;
      s.lastAtlasSnapshotId = snapshotId ?? null;
    }),
  setDashboardMode: (mode) =>
    set((s) => {
      s.dashboardMode = mode;
    }),
});
