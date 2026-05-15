import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, LifecycleSlice } from "./types";

export const createLifecycleSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  LifecycleSlice
> = (set, get) => ({
  lifecycleSummary: null,
  isLoadingLifecycleSummary: false,

  loadLifecycleSummary: async () => {
    if (get().isLoadingLifecycleSummary) return;
    set((s) => {
      s.isLoadingLifecycleSummary = true;
    });
    try {
      const data = await api.getLifecycleSummary();
      set((s) => {
        s.lifecycleSummary = data;
        s.isLoadingLifecycleSummary = false;
      });
    } catch {
      set((s) => {
        s.isLoadingLifecycleSummary = false;
      });
    }
  },
});
