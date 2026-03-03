import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, SelectionSlice } from "./types";

export const createSelectionSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  SelectionSlice
> = (set) => ({
  selectedNodeId: null,
  hoveredNodeId: null,
  selectedEdgeId: null,
  searchQuery: "",
  searchResults: [],
  isSearching: false,

  selectNode: (nodeId) =>
    set((s) => {
      s.selectedNodeId = nodeId;
    }),
  hoverNode: (nodeId) =>
    set((s) => {
      s.hoveredNodeId = nodeId;
    }),
  selectEdge: (edgeId) =>
    set((s) => {
      s.selectedEdgeId = edgeId;
    }),
  setSearchQuery: (query) =>
    set((s) => {
      s.searchQuery = query;
    }),

  executeSearch: async (query: string) => {
    if (!query.trim()) {
      set((s) => {
        s.searchResults = [];
        s.isSearching = false;
      });
      return;
    }
    set((s) => {
      s.isSearching = true;
    });
    try {
      const results = await api.searchEntities({ q: query, limit: 20 });
      set((s) => {
        s.searchResults = results;
        s.isSearching = false;
      });
    } catch {
      set((s) => {
        s.isSearching = false;
      });
    }
  },

  clearSearch: () =>
    set((s) => {
      s.searchQuery = "";
      s.searchResults = [];
      s.isSearching = false;
    }),
});
