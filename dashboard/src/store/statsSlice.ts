import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, StatsSlice } from "./types";

export const createStatsSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  StatsSlice
> = (set, get) => ({
  stats: null,
  storage: null,
  isLoadingStats: false,
  isLoadingStorage: false,

  loadStats: async () => {
    if (get().isLoadingStats) return;
    set((s) => {
      s.isLoadingStats = true;
    });
    try {
      const data = await api.getStats();
      set((s) => {
        s.stats = data;
        s.isLoadingStats = false;
      });
    } catch {
      set((s) => {
        s.isLoadingStats = false;
      });
    }
  },

  loadStorage: async () => {
    if (get().isLoadingStorage) return;
    set((s) => {
      s.isLoadingStorage = true;
    });
    try {
      const data = await api.getStorage();
      set((s) => {
        s.storage = data;
        s.isLoadingStorage = false;
      });
    } catch {
      set((s) => {
        s.isLoadingStorage = false;
      });
    }
  },
});
