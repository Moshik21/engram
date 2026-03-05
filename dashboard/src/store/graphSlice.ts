import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, GraphSlice, GraphDelta } from "./types";

export const createGraphSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  GraphSlice
> = (set, get) => ({
  nodes: {},
  edges: {},
  centerNodeId: null,
  isLoading: false,
  error: null,

  loadNeighborhood: async (centerId?: string, depth?: number) => {
    set((s) => {
      s.isLoading = true;
      s.error = null;
    });
    try {
      const data = await api.getNeighborhood({
        center: centerId,
        depth,
        maxNodes: get().graphMaxNodes,
      });
      set((s) => {
        s.nodes = {};
        s.edges = {};
        for (const n of data.nodes) s.nodes[n.id] = n;
        for (const e of data.edges) s.edges[e.id] = e;
        s.centerNodeId = data.centerId;
        s.isLoading = false;
      });
    } catch (e) {
      set((s) => {
        s.error = (e as Error).message;
        s.isLoading = false;
      });
    }
  },

  loadInitialGraph: async () => {
    await get().loadNeighborhood(undefined, 3);
  },

  expandNode: async (nodeId: string) => {
    try {
      const data = await api.getNeighbors(nodeId, { depth: 1, maxNodes: Math.min(100, get().graphMaxNodes) });
      set((s) => {
        for (const n of data.nodes) {
          if (!s.nodes[n.id]) s.nodes[n.id] = n;
        }
        for (const e of data.edges) {
          if (!s.edges[e.id]) s.edges[e.id] = e;
        }
      });
    } catch (e) {
      set((s) => {
        s.error = (e as Error).message;
      });
    }
  },

  mergeGraphDelta: (delta: GraphDelta) =>
    set((s) => {
      if (delta.nodesAdded) {
        for (const n of delta.nodesAdded) s.nodes[n.id] = n;
      }
      if (delta.nodesUpdated) {
        for (const n of delta.nodesUpdated) {
          if (s.nodes[n.id]) s.nodes[n.id] = n;
        }
      }
      if (delta.nodesRemoved) {
        for (const id of delta.nodesRemoved) delete s.nodes[id];
      }
      if (delta.edgesAdded) {
        for (const e of delta.edgesAdded) s.edges[e.id] = e;
      }
      if (delta.edgesRemoved) {
        for (const id of delta.edgesRemoved) delete s.edges[id];
      }
    }),

  updateNodeActivations: (updates) =>
    set((s) => {
      for (const { entityId, activation } of updates) {
        if (s.nodes[entityId]) {
          s.nodes[entityId].activationCurrent = activation;
        }
      }
    }),

  loadGraphAt: async (timestamp: string, centerId?: string) => {
    set((s) => {
      s.isLoading = true;
      s.error = null;
    });
    try {
      const center = centerId || get().centerNodeId || undefined;
      const data = await api.getGraphAt(timestamp, center);
      set((s) => {
        s.nodes = {};
        s.edges = {};
        for (const n of data.nodes) s.nodes[n.id] = n;
        for (const e of data.edges) s.edges[e.id] = e;
        s.centerNodeId = data.centerId;
        s.isLoading = false;
      });
    } catch (e) {
      set((s) => {
        s.error = (e as Error).message;
        s.isLoading = false;
      });
    }
  },

  clear: () =>
    set((s) => {
      s.nodes = {};
      s.edges = {};
      s.centerNodeId = null;
      s.error = null;
    }),
});
