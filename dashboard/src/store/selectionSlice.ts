import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, SelectionSlice } from "./types";

const MAX_ENTITY_HISTORY = 50;

export const createSelectionSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  SelectionSlice
> = (set, get) => ({
  selectedNodeId: null,
  hoveredNodeId: null,
  selectedEdgeId: null,
  searchQuery: "",
  searchResults: [],
  isSearching: false,
  searchError: null,
  entityHistory: [],
  entityHistoryIndex: -1,
  isNavigatingHistory: false,

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
        s.searchError = null;
      });
      return;
    }
    set((s) => {
      s.isSearching = true;
      s.searchError = null;
    });
    try {
      const results = await api.searchEntities({ q: query, limit: 20 });
      set((s) => {
        s.searchResults = results;
        s.isSearching = false;
        s.searchError = null;
      });
    } catch (err) {
      set((s) => {
        s.isSearching = false;
        s.searchError = err instanceof Error ? err.message : "Search failed";
      });
    }
  },

  clearSearch: () =>
    set((s) => {
      s.searchQuery = "";
      s.searchResults = [];
      s.isSearching = false;
      s.searchError = null;
    }),

  pushEntityHistory: (nodeId: string) => {
    const state = get();
    if (state.isNavigatingHistory) return;
    set((s) => {
      // Truncate forward history
      s.entityHistory = s.entityHistory.slice(0, s.entityHistoryIndex + 1);
      // Don't push duplicates
      if (s.entityHistory[s.entityHistory.length - 1] !== nodeId) {
        s.entityHistory.push(nodeId);
        // Cap at max
        if (s.entityHistory.length > MAX_ENTITY_HISTORY) {
          s.entityHistory = s.entityHistory.slice(-MAX_ENTITY_HISTORY);
        }
      }
      s.entityHistoryIndex = s.entityHistory.length - 1;
    });
  },

  entityGoBack: () => {
    const state = get();
    if (state.entityHistoryIndex <= 0) return null;
    const newIndex = state.entityHistoryIndex - 1;
    const nodeId = state.entityHistory[newIndex];
    set((s) => {
      s.isNavigatingHistory = true;
      s.entityHistoryIndex = newIndex;
      s.selectedNodeId = nodeId;
    });
    // Reset guard after microtask
    setTimeout(() => set((s) => { s.isNavigatingHistory = false; }), 0);
    return nodeId;
  },

  entityGoForward: () => {
    const state = get();
    if (state.entityHistoryIndex >= state.entityHistory.length - 1) return null;
    const newIndex = state.entityHistoryIndex + 1;
    const nodeId = state.entityHistory[newIndex];
    set((s) => {
      s.isNavigatingHistory = true;
      s.entityHistoryIndex = newIndex;
      s.selectedNodeId = nodeId;
    });
    setTimeout(() => set((s) => { s.isNavigatingHistory = false; }), 0);
    return nodeId;
  },

  canEntityGoBack: () => {
    const state = get();
    return state.entityHistoryIndex > 0;
  },

  canEntityGoForward: () => {
    const state = get();
    return state.entityHistoryIndex < state.entityHistory.length - 1;
  },
});
