import type { StateCreator } from "zustand";
import type { EngramStore, TimeSlice } from "./types";

export const createTimeSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  TimeSlice
> = (set) => ({
  timePosition: null,
  timeRange: null,
  isTimeScrubbing: false,
  atlasSnapshotId: null,
  atlasHistory: [],

  setTimePosition: (ts) =>
    set((s) => {
      s.timePosition = ts;
    }),
  setTimeRange: (range) =>
    set((s) => {
      s.timeRange = range;
    }),
  setIsTimeScrubbing: (v) =>
    set((s) => {
      s.isTimeScrubbing = v;
    }),
  setAtlasSnapshotId: (snapshotId) =>
    set((s) => {
      s.atlasSnapshotId = snapshotId;
    }),
  setAtlasHistory: (history) =>
    set((s) => {
      s.atlasHistory = history;
    }),
});
