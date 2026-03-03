import type { StateCreator } from "zustand";
import type { EngramStore, WebSocketSlice } from "./types";

export const createWsSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  WebSocketSlice
> = (set) => ({
  readyState: "disconnected",
  lastSeq: 0,
  reconnectAttempt: 0,

  setReadyState: (state) =>
    set((s) => {
      s.readyState = state;
    }),
  setLastSeq: (seq) =>
    set((s) => {
      s.lastSeq = seq;
    }),
  setReconnectAttempt: (n) =>
    set((s) => {
      s.reconnectAttempt = n;
    }),
});
