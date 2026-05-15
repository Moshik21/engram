import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, EvaluationSlice } from "./types";

export const createEvaluationSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  EvaluationSlice
> = (set, get) => ({
  evaluationReport: null,
  isLoadingEvaluationReport: false,
  isSavingRecallEvaluation: false,
  isSavingSessionEvaluation: false,

  loadEvaluationReport: async () => {
    if (get().isLoadingEvaluationReport) return;
    set((s) => {
      s.isLoadingEvaluationReport = true;
    });
    try {
      const report = await api.getEvaluationReport();
      set((s) => {
        s.evaluationReport = report;
        s.isLoadingEvaluationReport = false;
      });
    } catch {
      set((s) => {
        s.isLoadingEvaluationReport = false;
      });
    }
  },

  recordRecallEvaluation: async (input) => {
    if (get().isSavingRecallEvaluation) return;
    set((s) => {
      s.isSavingRecallEvaluation = true;
    });
    try {
      await api.recordRecallEvaluation({ ...input, source: input.source ?? "dashboard" });
      await get().loadEvaluationReport();
    } finally {
      set((s) => {
        s.isSavingRecallEvaluation = false;
      });
    }
  },

  recordSessionContinuityEvaluation: async (input) => {
    if (get().isSavingSessionEvaluation) return;
    set((s) => {
      s.isSavingSessionEvaluation = true;
    });
    try {
      await api.recordSessionContinuityEvaluation({ ...input, source: input.source ?? "dashboard" });
      await get().loadEvaluationReport();
    } finally {
      set((s) => {
        s.isSavingSessionEvaluation = false;
      });
    }
  },
});
