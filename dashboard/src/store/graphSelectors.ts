import { useEffect, useMemo, useRef, useState } from "react";
import { useEngramStore } from "./index";
import { useShallow } from "zustand/shallow";

/**
 * Structural-only selector for graph nodes/edges.
 * Only triggers re-render when nodes/edges are added or removed,
 * NOT when activation values change.
 */
export function useGraphStructure() {
  return useEngramStore(
    useShallow((s) => {
      const nodeIds = Object.keys(s.nodes);
      const edgeIds = Object.keys(s.edges);
      return { nodeIds, edgeIds };
    }),
  );
}

/**
 * Returns node count without subscribing to the full nodes object.
 * Only triggers re-render when count changes.
 */
export function useNodeCount(): number {
  return useEngramStore((s) => Object.keys(s.nodes).length);
}

/**
 * Returns edge count without subscribing to the full edges object.
 */
export function useEdgeCount(): number {
  return useEngramStore((s) => Object.keys(s.edges).length);
}

/**
 * Returns a single node by ID. Only re-renders when that specific node changes.
 */
export function useNodeById(nodeId: string | null) {
  return useEngramStore((s) => (nodeId ? s.nodes[nodeId] ?? null : null));
}

/**
 * Returns a stable ref containing the latest activation values.
 * Subscribes outside the React render cycle — activation-only WebSocket
 * updates do NOT trigger React re-renders.
 *
 * Animation loop reads from this ref directly.
 */
export function useActivationRef(): React.MutableRefObject<Map<string, number>> {
  const ref = useRef(new Map<string, number>());

  useEffect(() => {
    // Initialize from current state
    const state = useEngramStore.getState();
    const map = new Map<string, number>();
    for (const [id, node] of Object.entries(state.nodes)) {
      map.set(id, node.activationCurrent);
    }
    ref.current = map;

    // Subscribe to store changes, update ref without triggering render
    const unsub = useEngramStore.subscribe((state) => {
      const newMap = new Map<string, number>();
      for (const [id, node] of Object.entries(state.nodes)) {
        newMap.set(id, node.activationCurrent);
      }
      ref.current = newMap;
    });

    return unsub;
  }, []);

  return ref;
}

/**
 * Returns a stable ref containing the latest node data (for position reading).
 * Subscribes outside React render cycle.
 */
export function useNodeDataRef(): React.MutableRefObject<
  Record<string, { entityType: string; accessCount: number; activationCurrent: number }>
> {
  const ref = useRef<Record<string, { entityType: string; accessCount: number; activationCurrent: number }>>({});

  useEffect(() => {
    const update = () => {
      const state = useEngramStore.getState();
      const data: Record<string, { entityType: string; accessCount: number; activationCurrent: number }> = {};
      for (const [id, node] of Object.entries(state.nodes)) {
        data[id] = {
          entityType: node.entityType,
          accessCount: node.accessCount,
          activationCurrent: node.activationCurrent,
        };
      }
      ref.current = data;
    };

    update();
    const unsub = useEngramStore.subscribe(update);
    return unsub;
  }, []);

  return ref;
}

/** Keys written by the d3-force simulation that must survive across rebuilds. */
const SIM_KEYS = ["x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "__threeObj"] as const;

/**
 * Structural fingerprint that only changes when nodes/edges are added or removed.
 * Uses a store subscription + ref to avoid re-rendering on activation changes.
 */
function useStructuralFingerprint() {
  const prevNodeFp = useRef("");
  const prevEdgeFp = useRef("");
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const unsub = useEngramStore.subscribe((state) => {
      const nodeFp = Object.keys(state.nodes).sort().join(",");
      const edgeFp = Object.keys(state.edges).sort().join(",");
      if (nodeFp !== prevNodeFp.current || edgeFp !== prevEdgeFp.current) {
        prevNodeFp.current = nodeFp;
        prevEdgeFp.current = edgeFp;
        setVersion((v) => v + 1);
      }
    });
    // Initialize
    const state = useEngramStore.getState();
    prevNodeFp.current = Object.keys(state.nodes).sort().join(",");
    prevEdgeFp.current = Object.keys(state.edges).sort().join(",");
    return unsub;
  }, []);

  return version;
}

/**
 * Graph data memo that only depends on structural changes (node/edge add/remove),
 * not on activation value changes.
 *
 * Uses a stable object cache so the force simulation's position data (x/y/z/vx/vy/vz)
 * and Three.js object refs (__threeObj) survive across rebuilds. New nodes appear near
 * a connected neighbor instead of at a random position.
 */
export function useStableGraphData() {
  const structureVersion = useStructuralFingerprint();

  // Stable caches — persist across renders so simulation-written fields survive
  const nodeCacheRef = useRef<Map<string, Record<string, unknown>>>(new Map());
  const edgeCacheRef = useRef<Map<string, Record<string, unknown>>>(new Map());

  return useMemo(() => {
    const nodes = useEngramStore.getState().nodes;
    const edges = useEngramStore.getState().edges;
    const nodeCache = nodeCacheRef.current;
    const edgeCache = edgeCacheRef.current;
    const currentNodeIds = new Set(Object.keys(nodes));
    const currentEdgeIds = new Set(Object.keys(edges));
    const newNodeIds: string[] = [];

    // ── Nodes: update existing, create new, remove stale ──
    for (const [id, storeNode] of Object.entries(nodes)) {
      const existing = nodeCache.get(id);
      if (existing) {
        // Preserve simulation keys, update everything else from store
        const saved: Record<string, unknown> = {};
        for (const key of SIM_KEYS) {
          if (key in existing) saved[key] = existing[key];
        }
        Object.assign(existing, storeNode, saved);
      } else {
        const fresh: Record<string, unknown> = { ...storeNode, __isNew: true };
        nodeCache.set(id, fresh);
        newNodeIds.push(id);
      }
    }

    // Remove nodes no longer in store
    for (const id of nodeCache.keys()) {
      if (!currentNodeIds.has(id)) nodeCache.delete(id);
    }

    // ── Edges: update existing (preserve __curveRotation), create new ──
    for (const [id, storeEdge] of Object.entries(edges)) {
      const existing = edgeCache.get(id);
      if (existing) {
        const savedRotation = existing.__curveRotation;
        Object.assign(existing, storeEdge);
        existing.__curveRotation = savedRotation;
      } else {
        edgeCache.set(id, {
          ...storeEdge,
          __curveRotation: Math.random() * Math.PI * 2,
        });
      }
    }

    // Remove edges no longer in store
    for (const id of edgeCache.keys()) {
      if (!currentEdgeIds.has(id)) edgeCache.delete(id);
    }

    // ── Position new nodes near a connected neighbor ──
    const edgeList = Array.from(edgeCache.values());
    for (const newId of newNodeIds) {
      const newNode = nodeCache.get(newId);
      if (!newNode) continue;

      // Find a neighbor that already has a position
      let placed = false;
      for (const edge of edgeList) {
        const src = typeof edge.source === "object" && edge.source !== null
          ? (edge.source as Record<string, unknown>).id as string
          : edge.source as string;
        const tgt = typeof edge.target === "object" && edge.target !== null
          ? (edge.target as Record<string, unknown>).id as string
          : edge.target as string;

        const neighborId = src === newId ? tgt : tgt === newId ? src : null;
        if (!neighborId) continue;

        const neighbor = nodeCache.get(neighborId);
        if (neighbor && typeof neighbor.x === "number") {
          const jitter = () => (Math.random() - 0.5) * 40;
          newNode.x = (neighbor.x as number) + jitter();
          newNode.y = (neighbor.y as number) + jitter();
          newNode.z = (neighbor.z as number) + jitter();
          placed = true;
          break;
        }
      }

      // If no positioned neighbor found, leave undefined (d3 will place it)
      delete newNode.__isNew;
      if (!placed) {
        // no-op: d3-force will assign a random position
      }
    }

    const nodeList = Array.from(nodeCache.values());
    return { nodes: nodeList, links: edgeList, _newCount: newNodeIds.length };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structureVersion]);
}
