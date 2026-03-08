import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type {
  EngramStore,
  GraphSlice,
  GraphDelta,
  GraphRepresentationMeta,
} from "./types";

function buildNeighborhoodRepresentation(
  scope: "neighborhood" | "temporal",
  nodeCount: number,
  edgeCount: number,
  totalInNeighborhood: number,
  truncated: boolean,
): GraphRepresentationMeta {
  return {
    scope,
    layout: "force",
    representedEntityCount: totalInNeighborhood,
    representedEdgeCount: edgeCount,
    displayedNodeCount: nodeCount,
    displayedEdgeCount: edgeCount,
    truncated,
  };
}

export const createGraphSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  GraphSlice
> = (set, get) => ({
  nodes: {},
  edges: {},
  centerNodeId: null,
  brainMapScope: "atlas",
  representation: null,
  atlas: null,
  activeRegionId: null,
  regionData: null,
  isLoading: false,
  error: null,

  loadAtlas: async (options?: { refresh?: boolean; snapshotId?: string | null }) => {
    set((s) => {
      s.isLoading = true;
      s.error = null;
    });
    try {
      const [data, historyResponse] = await Promise.all([
        api.getGraphAtlas({
          refresh: options?.refresh,
          snapshotId: options?.snapshotId,
        }),
        api.getGraphAtlasHistory({ limit: 48 }),
      ]);
      set((s) => {
        s.atlas = data;
        s.atlasHistory = historyResponse.items;
        s.atlasSnapshotId = options?.snapshotId ?? null;
        s.representation = data.representation;
        s.brainMapScope = "atlas";
        s.activeRegionId = null;
        s.regionData = null;
        s.centerNodeId = null;
        s.isLoading = false;
      });
      get().selectNode(null);
      get().hoverNode(null);
      get().selectEdge(null);
    } catch (e) {
      set((s) => {
        s.error = (e as Error).message;
        s.isLoading = false;
      });
    }
  },

  loadRegion: async (
    regionId: string,
    options?: { refresh?: boolean; snapshotId?: string | null },
  ) => {
    set((s) => {
      s.isLoading = true;
      s.error = null;
    });
    try {
      const [data, historyResponse] = await Promise.all([
        api.getGraphRegion(regionId, {
          refresh: options?.refresh,
          snapshotId: options?.snapshotId,
        }),
        api.getGraphAtlasHistory({ limit: 48 }),
      ]);
      set((s) => {
        s.atlasHistory = historyResponse.items;
        s.atlasSnapshotId = options?.snapshotId ?? null;
        s.regionData = data;
        s.activeRegionId = data.region.id;
        s.representation = data.representation;
        s.brainMapScope = "region";
        s.centerNodeId = null;
        s.isLoading = false;
      });
      get().selectNode(null);
      get().hoverNode(null);
      get().selectEdge(null);
    } catch (e) {
      if (
        options?.snapshotId &&
        (e as Error).message.includes("404")
      ) {
        await get().loadAtlas({ snapshotId: options.snapshotId });
        return;
      }
      set((s) => {
        s.error = (e as Error).message;
        s.isLoading = false;
      });
    }
  },

  loadNeighborhood: async (
    centerId?: string,
    depth?: number,
    options?: { regionId?: string | null },
  ) => {
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
        s.brainMapScope = "neighborhood";
        s.activeRegionId = options?.regionId ?? null;
        if (!options?.regionId) {
          s.regionData = null;
        }
        s.representation =
          data.representation ??
          buildNeighborhoodRepresentation(
            "neighborhood",
            data.nodes.length,
            data.edges.length,
            data.totalInNeighborhood,
            data.truncated,
          );
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
    const state = get();
    if (state.timePosition && state.centerNodeId) {
      await state.loadGraphAt(state.timePosition, state.centerNodeId);
      return;
    }
    if (state.brainMapScope === "region" && state.activeRegionId) {
      await state.loadRegion(state.activeRegionId, {
        snapshotId: state.atlasSnapshotId,
      });
      return;
    }
    if (state.brainMapScope === "atlas" && state.atlasSnapshotId) {
      await state.loadAtlas({ snapshotId: state.atlasSnapshotId });
      return;
    }
    if (state.brainMapScope !== "atlas" && state.centerNodeId) {
      await state.loadNeighborhood(state.centerNodeId, 2, {
        regionId: state.activeRegionId,
      });
      return;
    }
    await state.loadAtlas();
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
        s.brainMapScope = "neighborhood";
        s.representation =
          data.representation ??
          buildNeighborhoodRepresentation(
            "neighborhood",
            Object.keys(s.nodes).length,
            Object.keys(s.edges).length,
            data.totalInNeighborhood,
            data.truncated,
          );
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
      if (s.representation && s.representation.scope !== "atlas") {
        s.representation.displayedNodeCount = Object.keys(s.nodes).length;
        s.representation.displayedEdgeCount = Object.keys(s.edges).length;
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
        s.brainMapScope = "temporal";
        s.representation =
          data.representation ??
          buildNeighborhoodRepresentation(
            "temporal",
            data.nodes.length,
            data.edges.length,
            data.totalInNeighborhood,
            data.truncated,
          );
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
      s.brainMapScope = "atlas";
      s.representation = null;
      s.atlas = null;
      s.atlasSnapshotId = null;
      s.atlasHistory = [];
      s.activeRegionId = null;
      s.regionData = null;
      s.error = null;
    }),
});
