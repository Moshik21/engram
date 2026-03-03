import type { StateCreator } from "zustand";
import type { EngramStore, ActivationSlice } from "./types";
import { api } from "../api/client";

export const createActivationSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  ActivationSlice
> = (set) => ({
  activationLeaderboard: [],
  selectedActivationEntity: null,
  decayCurve: [],
  decayFormula: "",
  accessEvents: [],
  isActivationSubscribed: false,
  isLoadingCurve: false,

  setActivationLeaderboard: (items) =>
    set((s) => {
      s.activationLeaderboard = items;
    }),

  selectActivationEntity: (id) =>
    set((s) => {
      s.selectedActivationEntity = id;
    }),

  loadDecayCurve: async (entityId: string) => {
    set((s) => {
      s.isLoadingCurve = true;
    });
    try {
      const data = await api.getActivationCurve(entityId);
      set((s) => {
        s.decayCurve = data.curve;
        s.decayFormula = data.formula;
        s.accessEvents = data.accessEvents;
        s.isLoadingCurve = false;
      });
    } catch {
      set((s) => {
        s.isLoadingCurve = false;
      });
    }
  },

  setIsActivationSubscribed: (v) =>
    set((s) => {
      s.isActivationSubscribed = v;
    }),
});
