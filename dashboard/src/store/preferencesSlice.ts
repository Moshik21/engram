import type { StateCreator } from "zustand";
import type { EngramStore, PreferencesSlice } from "./types";

export const createPreferencesSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  PreferencesSlice
> = (set) => ({
  currentView: "graph",
  renderMode: "3d",
  showActivationHeatmap: true,
  showEdgeLabels: false,
  showFpsOverlay: false,
  darkMode: true,
  graphMaxNodes: 1500,
  lastAtlasVisitAt: null,
  lastAtlasSnapshotId: null,

  setCurrentView: (view) =>
    set((s) => {
      s.currentView = view;
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
});
