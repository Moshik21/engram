import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, ConsolidationSlice } from "./types";

export const createConsolidationSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  ConsolidationSlice
> = (set, get) => ({
  cycles: [],
  isLoadingCycles: false,
  selectedCycleId: null,
  cycleDetail: null,
  isLoadingDetail: false,
  isRunning: false,
  schedulerActive: false,
  pressure: null,

  loadStatus: async () => {
    try {
      const data = await api.getConsolidationStatus();
      set((s) => {
        s.isRunning = data.is_running;
        s.schedulerActive = data.scheduler_active;
        s.pressure = data.pressure ?? null;
      });
    } catch {
      // ignore
    }
  },

  loadCycles: async () => {
    if (get().isLoadingCycles) return;
    set((s) => {
      s.isLoadingCycles = true;
    });
    try {
      const data = await api.getConsolidationHistory();
      set((s) => {
        s.cycles = data.cycles;
        s.isLoadingCycles = false;
      });
    } catch {
      set((s) => {
        s.isLoadingCycles = false;
      });
    }
  },

  selectCycle: (id) => {
    set((s) => {
      s.selectedCycleId = id;
      if (!id) s.cycleDetail = null;
    });
    if (id) {
      get().loadCycleDetail(id);
    }
  },

  loadCycleDetail: async (id: string) => {
    set((s) => {
      s.isLoadingDetail = true;
    });
    try {
      const data = await api.getConsolidationCycle(id);
      set((s) => {
        s.cycleDetail = data;
        s.isLoadingDetail = false;
      });
    } catch {
      set((s) => {
        s.isLoadingDetail = false;
      });
    }
  },

  triggerCycle: async (dryRun: boolean) => {
    try {
      await api.triggerConsolidation(dryRun);
      set((s) => {
        s.isRunning = true;
      });
      // Refresh status after a delay to catch completion
      setTimeout(() => {
        get().loadStatus();
        get().loadCycles();
      }, 1000);
    } catch {
      // ignore
    }
  },
});
