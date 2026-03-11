import { useCallback, useRef, useEffect, useState, useMemo, lazy, Suspense } from "react";
import ForceGraph3D from "react-force-graph-3d";
import ForceGraph2D from "react-force-graph-2d";
import * as THREE from "three";
import { setupBloomPass, createAmbientParticles } from "./graph/NeuralSceneSetup";
import { useEngramStore } from "../store";
import { activationColor, entityColor } from "../lib/colors";
import { NodeTooltip } from "./NodeTooltip";
import { EmptyState } from "./EmptyState";
import { Stats } from "../lib/stats";

const StressTestPanel = lazy(() => import("./StressTestPanel"));

// Tier system
import { TierClassifier, type Tier } from "./graph/TierClassifier";
import { FocusTierRenderer } from "./graph/FocusTierRenderer";
import { ActiveTierRenderer } from "./graph/ActiveTierRenderer";
import { DormantTierRenderer } from "./graph/DormantTierRenderer";
import { SharedResources } from "./graph/SharedResources";
import { AdjacencyMap } from "./graph/AdjacencyMap";
import { EdgeLineRenderer } from "./graph/EdgeLineRenderer";
import { AnimationBudget } from "./graph/AnimationBudget";
import { DisposalRegistry } from "./graph/DisposalRegistry";
import { LODController } from "./graph/LODController";
import { useStableGraphData, useActivationRef } from "../store/graphSelectors";

// Navigation
import { useNavigationHistory, type CameraBookmark } from "./graph/useNavigationHistory";
import { useKeyboardNavigation } from "./graph/useKeyboardNavigation";
import { NavigationBreadcrumbs } from "./graph/NavigationBreadcrumbs";
import { KeyboardHelpOverlay } from "./graph/KeyboardHelpOverlay";

const PULSE_DURATION_MS = 1500;
const CASCADE_DELAY_MS = 300;
const CASCADE_INTENSITY = 0.6;
const CASCADE_MAX_QUEUE = 100;

interface CascadeEntry {
  targetId: string;
  fireAt: number;
  intensity: number;
}

type FgRef = {
  d3Force: (name: string) => unknown;
  emitParticle?: (link: unknown) => void;
  scene?: () => THREE.Scene;
  renderer?: () => THREE.WebGLRenderer;
  camera?: () => THREE.Camera;
  postProcessingComposer?: () => unknown;
  cameraPosition?: (
    pos: { x: number; y: number; z: number },
    lookAt?: { x: number; y: number; z: number },
    transitionMs?: number,
  ) => void;
  zoomToFit?: (durationMs?: number, padding?: number, nodeFilter?: (node: unknown) => boolean) => void;
} | null;

const FLY_TO_DURATION_MS = 800;
const FLY_TO_OFFSET_Z = 120;

export function GraphExplorer() {
  const isLoading = useEngramStore((s) => s.isLoading);
  const renderMode = useEngramStore((s) => s.renderMode);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const showEdgeLabels = useEngramStore((s) => s.showEdgeLabels);
  const showFpsOverlay = useEngramStore((s) => s.showFpsOverlay);
  const toggleFpsOverlay = useEngramStore((s) => s.toggleFpsOverlay);
  const selectNode = useEngramStore((s) => s.selectNode);
  const hoverNode = useEngramStore((s) => s.hoverNode);
  const expandNode = useEngramStore((s) => s.expandNode);

  const setRenderMode = useEngramStore((s) => s.setRenderMode);
  const toggleHeatmap = useEngramStore((s) => s.toggleActivationHeatmap);
  const toggleEdgeLabelsAction = useEngramStore((s) => s.toggleEdgeLabels);
  const setSearchOverlayOpen = useEngramStore((s) => s.setSearchOverlayOpen);
  const nodes = useEngramStore((s) => s.nodes);

  const [showStressTest, setShowStressTest] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [breadcrumbTick, setBreadcrumbTick] = useState(0);
  const fgRef = useRef<FgRef>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const graphData = useStableGraphData();
  const activationRef = useActivationRef();
  const navHistory = useNavigationHistory();
  // Tracks the last selectedNodeId that was handled by direct click (to avoid double fly-to)
  const prevSelectedRef = useRef<string | null>(null);

  // ── Stats.js FPS overlay ──
  const statsRef = useRef<Stats | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    if (showFpsOverlay) {
      const stats = new Stats();
      container.appendChild(stats.dom);
      statsRef.current = stats;
      return () => {
        container.removeChild(stats.dom);
        statsRef.current = null;
      };
    } else {
      if (statsRef.current) {
        container.removeChild(statsRef.current.dom);
        statsRef.current = null;
      }
    }
  }, [showFpsOverlay]);

  // ── 2D mode passive FPS loop ──
  useEffect(() => {
    if (renderMode !== "2d" || !showFpsOverlay) return;
    let rafId: number;
    const tick = () => {
      statsRef.current?.begin();
      statsRef.current?.end();
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [renderMode, showFpsOverlay]);

  // ── Camera fly-to helper ──
  const flyToNode = useCallback(
    (nodeId: string) => {
      const fg = fgRef.current;
      if (!fg?.cameraPosition) return;
      // Find node position from graphData (d3-force attaches x/y/z)
      const node = graphData.nodes.find(
        (n: Record<string, unknown>) => n.id === nodeId,
      ) as Record<string, unknown> | undefined;
      if (!node || node.x == null || node.y == null) return;
      const x = node.x as number;
      const y = node.y as number;
      const z = (node.z as number) ?? 0;
      const lookAt = { x, y, z };
      const position = { x, y, z: z + FLY_TO_OFFSET_Z };
      fg.cameraPosition(position, lookAt, FLY_TO_DURATION_MS);
      return { position, lookAt };
    },
    [graphData.nodes],
  );

  const flyToBookmark = useCallback(
    (pos: CameraBookmark["position"], lookAt: CameraBookmark["lookAt"]) => {
      const fg = fgRef.current;
      if (!fg?.cameraPosition) return;
      fg.cameraPosition(pos, lookAt, FLY_TO_DURATION_MS);
    },
    [],
  );

  const nodeExists = useCallback(
    (id: string) => id in nodes,
    [nodes],
  );

  // Node name lookup for breadcrumbs
  const nodeNames = useMemo(() => {
    const map = new Map<string, string>();
    for (const n of graphData.nodes) {
      const node = n as Record<string, unknown>;
      if (typeof node.id === "string" && typeof node.name === "string") {
        map.set(node.id, node.name);
      }
    }
    return map;
  }, [graphData.nodes]);

  // ── Keyboard shortcuts (Shift+F, Shift+T stay here; rest in useKeyboardNavigation) ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.shiftKey && e.key === "F") {
        e.preventDefault();
        toggleFpsOverlay();
      }
      if (import.meta.env.DEV && e.shiftKey && e.key === "T") {
        e.preventDefault();
        setShowStressTest((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [toggleFpsOverlay]);

  // ── Navigation keyboard shortcuts ──
  useKeyboardNavigation({
    openSearch: () => setSearchOverlayOpen(true),
    deselect: () => {
      selectNode(null);
      setShowHelp(false);
    },
    centerOnSelected: () => {
      const selectedId = useEngramStore.getState().selectedNodeId;
      if (selectedId) flyToNode(selectedId);
    },
    goBack: () => {
      const nodeId = navHistory.goBack(flyToBookmark, nodeExists);
      if (nodeId) selectNode(nodeId);
    },
    goForward: () => {
      const nodeId = navHistory.goForward(flyToBookmark, nodeExists);
      if (nodeId) selectNode(nodeId);
    },
    resetCamera: () => {
      const fg = fgRef.current;
      if (fg?.zoomToFit) fg.zoomToFit(FLY_TO_DURATION_MS, 40);
    },
    toggleHeatmap,
    toggleEdgeLabels: toggleEdgeLabelsAction,
    toggleRenderMode: () => {
      const current = useEngramStore.getState().renderMode;
      setRenderMode(current === "3d" ? "2d" : "3d");
    },
    showHelp: () => setShowHelp((v) => !v),
    expandSelected: () => {
      const selectedId = useEngramStore.getState().selectedNodeId;
      if (selectedId) expandNode(selectedId);
    },
  });

  // ── Tier system refs (stable across renders) ──
  const sharedRef = useRef<SharedResources | null>(null);
  const registryRef = useRef<DisposalRegistry | null>(null);
  const classifierRef = useRef<TierClassifier | null>(null);
  const focusRendererRef = useRef<FocusTierRenderer | null>(null);
  const activeRendererRef = useRef<ActiveTierRenderer | null>(null);
  const dormantRendererRef = useRef<DormantTierRenderer | null>(null);
  const adjacencyRef = useRef<AdjacencyMap | null>(null);
  const budgetRef = useRef<AnimationBudget | null>(null);
  const lodRef = useRef<LODController | null>(null);
  const edgeRendererRef = useRef<EdgeLineRenderer | null>(null);
  const batchRenderersAddedRef = useRef(false);

  // Track which nodes are in which tier for batch renderers
  const nodeTierMapRef = useRef<Map<string, Tier>>(new Map());

  // Cache Three.js node objects — only recreate when tier changes
  const nodeObjectCacheRef = useRef<Map<string, { tier: Tier; obj: THREE.Group }>>(new Map());

  // Initialize tier system
  useEffect(() => {
    const shared = new SharedResources();
    const registry = new DisposalRegistry();
    const nodeObjectCache = nodeObjectCacheRef.current;
    sharedRef.current = shared;
    registryRef.current = registry;
    classifierRef.current = new TierClassifier();
    focusRendererRef.current = new FocusTierRenderer(shared, registry);
    activeRendererRef.current = new ActiveTierRenderer(shared);
    dormantRendererRef.current = new DormantTierRenderer();
    adjacencyRef.current = new AdjacencyMap();
    budgetRef.current = new AnimationBudget();
    lodRef.current = new LODController();
    edgeRendererRef.current = new EdgeLineRenderer();

    return () => {
      registry.disposeAll();
      shared.dispose();
      activeRendererRef.current?.dispose();
      dormantRendererRef.current?.dispose();
      edgeRendererRef.current?.dispose();
      classifierRef.current?.clear();
      adjacencyRef.current?.clear();
      budgetRef.current?.reset();
      nodeObjectCache.clear();
      batchRenderersAddedRef.current = false;
    };
  }, []);

  // Rebuild adjacency map + edge renderer + invalidate LOD visibility when graph structure changes
  useEffect(() => {
    if (adjacencyRef.current && graphData.links) {
      adjacencyRef.current.rebuild(graphData.links);
    }
    if (edgeRendererRef.current && graphData.links) {
      edgeRendererRef.current.rebuild(
        graphData.links as Array<{
          source?: string | { id?: string };
          target?: string | { id?: string };
        }>,
      );
    }
    lodRef.current?.invalidateVisibility();
  }, [graphData]);

  // Add batch renderers + edge renderer to scene once fg.scene() is available
  useEffect(() => {
    if (renderMode !== "3d") return;
    let attachedScene: THREE.Scene | null = null;

    const addToScene = () => {
      const fg = fgRef.current as { scene?: () => THREE.Scene } | null;
      if (!fg?.scene) return false;
      if (batchRenderersAddedRef.current) return true;

      const scene = fg.scene();
      attachedScene = scene;
      if (activeRendererRef.current) scene.add(activeRendererRef.current.mesh);
      if (dormantRendererRef.current) scene.add(dormantRendererRef.current.points);
      if (edgeRendererRef.current) scene.add(edgeRendererRef.current.lineSegments);
      batchRenderersAddedRef.current = true;
      return true;
    };

    if (!addToScene()) {
      const id = setInterval(() => {
        if (addToScene()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }

    return () => {
      if (attachedScene && batchRenderersAddedRef.current) {
        if (activeRendererRef.current) attachedScene.remove(activeRendererRef.current.mesh);
        if (dormantRendererRef.current) attachedScene.remove(dormantRendererRef.current.points);
        if (edgeRendererRef.current) attachedScene.remove(edgeRendererRef.current.lineSegments);
        batchRenderersAddedRef.current = false;
      }
    };
  }, [renderMode]);

  // Track previous node count for controlled reheat
  const prevNodeCountRef = useRef(0);

  // Tune force simulation + controlled reheat for incremental updates
  useEffect(() => {
    if (fgRef.current && renderMode === "3d") {
      const fg = fgRef.current as {
        d3Force: (
          name: string,
        ) =>
          | {
              strength?: (v: number) => unknown;
              distanceMax?: (v: number) => unknown;
              alpha?: (v: number) => unknown;
            }
          | undefined;
        d3ReheatSimulation?: () => void;
      };
      const charge = fg.d3Force("charge");
      if (charge?.strength && charge?.distanceMax) {
        const nodeCount = graphData.nodes.length;
        if (nodeCount > 1000) {
          charge.strength(-20);
          charge.distanceMax(200);
        } else if (nodeCount > 500) {
          charge.strength(-30);
          charge.distanceMax(250);
        } else {
          charge.strength(-40);
          charge.distanceMax(300);
        }
      }

      // Controlled reheat: gentle nudge when new nodes arrive,
      // skip on initial load (prevCount === 0) to let full layout settle naturally
      const currentCount = graphData.nodes.length;
      const prevCount = prevNodeCountRef.current;
      if (prevCount > 0 && currentCount > prevCount && fg.d3ReheatSimulation) {
        // d3ReheatSimulation sets alpha to 1.0 — we immediately tamp it down
        // to a gentle 0.15 so existing nodes barely shift while new ones settle
        fg.d3ReheatSimulation();
        const sim = fg.d3Force("link") as { simulation?: () => { alpha: (v: number) => void } } | undefined;
        // Access the underlying d3 simulation via the force-graph internals
        if (sim?.simulation) {
          sim.simulation().alpha(0.15);
        }
      }
      prevNodeCountRef.current = currentCount;
    }
  }, [graphData, renderMode]);

  // ── UnrealBloomPass ──
  useEffect(() => {
    if (renderMode !== "3d") return;
    return setupBloomPass(fgRef as React.RefObject<FgRef>);
  }, [renderMode]);

  // ── Ambient neurotransmitter particles ──
  const ambientParticlesRef = useRef<ReturnType<typeof createAmbientParticles> | null>(null);

  useEffect(() => {
    if (renderMode !== "3d") return;
    const particles = createAmbientParticles(
      fgRef as React.RefObject<FgRef>,
      200,
      450,
    );
    ambientParticlesRef.current = particles;

    return () => {
      particles.cleanup();
      ambientParticlesRef.current = null;
    };
  }, [renderMode]);

  // ── Optimized animation loop with tier-aware rendering + LOD ──
  const dirtyNodesRef = useRef<Set<string>>(new Set());
  const emittedPulsesRef = useRef<Set<string>>(new Set());
  const cascadeQueueRef = useRef<Map<string, CascadeEntry>>(new Map());
  const cascadedPulsesRef = useRef<Set<string>>(new Set());
  const centroidFrameRef = useRef(0);

  useEffect(() => {
    if (renderMode !== "3d") return;

    let rafId: number;
    const startTime = Date.now();

    const animate = () => {
      rafId = requestAnimationFrame(animate);
      statsRef.current?.begin();

      const pulses = useEngramStore.getState().activationPulses;
      const now = Date.now();
      const time = now - startTime;
      const fg = fgRef.current as FgRef;

      // 1. Particle drift
      ambientParticlesRef.current?.result?.animate();

      const classifier = classifierRef.current;
      const focusRenderer = focusRendererRef.current;
      const adjacency = adjacencyRef.current;
      const budget = budgetRef.current;
      const lod = lodRef.current;

      if (!classifier || !focusRenderer || !adjacency || !budget || !lod) return;

      // Get scene and main graph group (contains node + link objects)
      if (!fg?.scene) return;
      const scene = fg.scene();
      const mainGroup = scene.children.find(
        (c: THREE.Object3D) => c.type === "Group" && c.children.length > 10,
      ) as THREE.Group | undefined;
      if (!mainGroup) return;

      // ── LOD: Update camera distance + zoom tier ──
      if (fg.camera) {
        const cam = fg.camera();
        const tierChanged = lod.updateCamera(cam.position.x, cam.position.y, cam.position.z, now);
        if (tierChanged && edgeRendererRef.current) {
          edgeRendererRef.current.setOpacity(lod.config.edgeOpacity);
        }
      }

      // Pre-build O(1) lookup map from scene node objects (__data has x,y,z from d3-force)
      const nodeById = new Map<string, Record<string, unknown>>();
      const allNodeIds: string[] = [];
      for (const child of mainGroup.children) {
        const data = (child as unknown as Record<string, unknown>).__data as Record<string, unknown> | undefined;
        if (!data || typeof data.id !== "string") continue;
        // Skip link objects (they have source/target, not entityType)
        if ("source" in data && "target" in data && !("entityType" in data)) continue;
        const id = data.id;
        // Attach __threeObj reference for pulse animations
        data.__threeObj = child;
        nodeById.set(id, data);
        allNodeIds.push(id);
      }

      // Update centroid every ~30 frames (not every frame)
      centroidFrameRef.current++;
      if (centroidFrameRef.current % 30 === 0) {
        lod.updateCentroid(
          allNodeIds.map((id) => nodeById.get(id)) as Array<{ x?: number; y?: number; z?: number }>,
        );
      }

      // Rebuild LOD visibility when tier changes or periodically
      const activations = activationRef.current;
      if (centroidFrameRef.current % 15 === 0 || lod.alpha < 1) {
        lod.rebuildVisibility(activations, allNodeIds);
      }

      // 2. Classify all nodes and manage tier transitions
      classifier.updateTransitions(now);

      const activePulseIds = new Set<string>();
      const focusNodeIds: string[] = [];

      // Reclassify nodes with LOD-aware detail level
      for (const [id, node] of nodeById) {
        const activation = activations.get(id) ?? 0;

        // LOD determines max detail level for this node
        const detailLevel = lod.getNodeDetailLevel(id, activation);
        const newTier = classifier.classify(id, activation, detailLevel);
        const oldTier = nodeTierMapRef.current.get(id);

        if (oldTier !== newTier) {
          nodeTierMapRef.current.set(id, newTier);
          // Remove from old batch renderer
          if (oldTier === "active") activeRendererRef.current?.removeNode(id);
          if (oldTier === "dormant") dormantRendererRef.current?.removeNode(id);
          // Add to new batch renderer (focus tier uses real meshes, not batched)
          if (newTier === "active") {
            activeRendererRef.current?.addNode(id, (node.entityType as string) ?? "Other");
          }
          if (newTier === "dormant") {
            dormantRendererRef.current?.addNode(id, (node.entityType as string) ?? "Other");
          }
          // Force nodeThreeObject rebuild when transitioning to/from focus
          if (oldTier === "focus" || newTier === "focus") {
            nodeObjectCacheRef.current.delete(id);
          }
        }

        if (newTier === "focus") focusNodeIds.push(id);
      }

      // Clean up nodes that were removed from graph — O(M) where M = tracked nodes
      for (const [id] of nodeTierMapRef.current) {
        if (!nodeById.has(id)) {
          registryRef.current?.disposeNode(id);
          classifier.removeNode(id);
          activeRendererRef.current?.removeNode(id);
          dormantRendererRef.current?.removeNode(id);
          nodeTierMapRef.current.delete(id);
          nodeObjectCacheRef.current.delete(id);
        }
      }

      // 4. Fire queued cascades (batched to avoid per-cascade store updates)
      const pendingCascades: Array<{ entityId: string; intensity: number }> = [];
      for (const [key, entry] of cascadeQueueRef.current) {
        if (now >= entry.fireAt) {
          cascadeQueueRef.current.delete(key);
          pendingCascades.push({ entityId: entry.targetId, intensity: entry.intensity });
        }
      }
      if (pendingCascades.length > 0) {
        const store = useEngramStore.getState();
        for (const c of pendingCascades) {
          store.addActivationPulse({
            entityId: c.entityId,
            name: "",
            entityType: "",
            activation: c.intensity,
            accessedVia: "cascade",
          });
        }
      }

      // 5. Process pulses (using adjacency map for O(P * degree))
      for (const pulse of pulses) {
        const intensity =
          pulse.cascadeIntensity !== undefined ? pulse.cascadeIntensity : 1;
        const elapsed = now - pulse.timestamp;
        if (elapsed > PULSE_DURATION_MS) continue;

        activePulseIds.add(pulse.entityId);
        dirtyNodesRef.current.add(pulse.entityId);

        const t = elapsed / PULSE_DURATION_MS;
        const easeOut = 1 - (1 - t) * (1 - t);
        const fadeIntensity = (1 - easeOut) * intensity;

        // Only animate pulse on Focus-tier nodes (they have full meshes)
        const nodeTier = nodeTierMapRef.current.get(pulse.entityId);
        if (nodeTier === "focus") {
          const graphNode = nodeById.get(pulse.entityId);
          if (graphNode) {
            const threeObj = graphNode.__threeObj as THREE.Group | undefined;
            if (threeObj && threeObj.userData.tier === "focus") {
              focusRenderer.animatePulse(threeObj, fadeIntensity);
            }
          }
        }

        // Emit edge particles + queue cascades using adjacency map
        const pulseKey = `${pulse.entityId}-${pulse.timestamp}`;
        if (!emittedPulsesRef.current.has(pulseKey) && fg.emitParticle) {
          emittedPulsesRef.current.add(pulseKey);

          // O(degree) instead of O(E)
          const neighbors = adjacency.getNeighbors(pulse.entityId);
          for (const { linkRef } of neighbors) {
            fg.emitParticle(linkRef);
          }

          // Queue 1-hop cascade (only from primary pulses)
          if (
            pulse.accessedVia !== "cascade" &&
            !cascadedPulsesRef.current.has(pulseKey) &&
            cascadeQueueRef.current.size < CASCADE_MAX_QUEUE
          ) {
            cascadedPulsesRef.current.add(pulseKey);
            for (const { neighborId } of neighbors) {
              if (cascadeQueueRef.current.size >= CASCADE_MAX_QUEUE) break;
              const cascadeKey = `${neighborId}-${pulse.timestamp}`;
              if (!cascadeQueueRef.current.has(cascadeKey)) {
                cascadeQueueRef.current.set(cascadeKey, {
                  targetId: neighborId,
                  fireAt: now + CASCADE_DELAY_MS,
                  intensity: CASCADE_INTENSITY,
                });
              }
            }
          }
        }
      }

      // 6. Reset dirty focus-tier nodes no longer pulsing
      for (const nodeId of dirtyNodesRef.current) {
        if (activePulseIds.has(nodeId)) continue;

        const nodeTier = nodeTierMapRef.current.get(nodeId);
        if (nodeTier === "focus") {
          const graphNode = nodeById.get(nodeId);
          if (graphNode) {
            const threeObj = graphNode.__threeObj as THREE.Group | undefined;
            if (threeObj && threeObj.userData.tier === "focus") {
              focusRenderer.resetPulse(threeObj);
            }
          }
        }
        dirtyNodesRef.current.delete(nodeId);
      }

      // 7. Budget-allocated breathing on Focus tier
      const pulsingIds = Array.from(activePulseIds);
      const animateIds = budget.allocate(pulsingIds, focusNodeIds);

      for (let i = 0; i < animateIds.length; i++) {
        const id = animateIds[i];
        if (activePulseIds.has(id)) continue; // Already animated by pulse

        const graphNode = nodeById.get(id);
        if (!graphNode) continue;

        const threeObj = graphNode.__threeObj as THREE.Group | undefined;
        if (threeObj && threeObj.userData.tier === "focus") {
          focusRenderer.animateBreathing(threeObj, time, i);
        }
      }

      // 8. Update batch renderer positions
      if (batchRenderersAddedRef.current) {
        activeRendererRef.current?.updatePositions(
          nodeById as Map<string, { x?: number; y?: number; z?: number }>,
          activations,
        );
        dormantRendererRef.current?.updatePositions(
          nodeById as Map<string, { x?: number; y?: number; z?: number }>,
        );
        edgeRendererRef.current?.updatePositions(nodeById);
      }

      // Housekeeping
      if (emittedPulsesRef.current.size > 200)
        emittedPulsesRef.current.clear();
      if (cascadedPulsesRef.current.size > 200)
        cascadedPulsesRef.current.clear();

      statsRef.current?.end();
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [renderMode, activationRef]);

  const handleNodeClick = useCallback(
    (node: { id?: string | number; x?: number; y?: number; z?: number }) => {
      const nodeId = node.id != null ? String(node.id) : null;
      // Mark as handled so the external selection effect doesn't double-fly
      prevSelectedRef.current = nodeId;
      selectNode(nodeId);
      if (nodeId && node.x != null && node.y != null) {
        const x = node.x;
        const y = node.y;
        const z = node.z ?? 0;
        const lookAt = { x, y, z };
        const position = { x, y, z: z + FLY_TO_OFFSET_Z };
        const fg = fgRef.current;
        if (fg?.cameraPosition) {
          fg.cameraPosition(position, lookAt, FLY_TO_DURATION_MS);
        }
        navHistory.push({ nodeId, position, lookAt });
        setBreadcrumbTick((t) => t + 1);
      }
    },
    [selectNode, navHistory],
  );
  const handleNodeRightClick = useCallback(
    (node: { id?: string | number; x?: number; y?: number; z?: number }) => {
      if (node.id != null) {
        const nodeId = String(node.id);
        prevSelectedRef.current = nodeId;
        expandNode(nodeId);
        if (node.x != null && node.y != null) {
          const x = node.x;
          const y = node.y;
          const z = node.z ?? 0;
          const lookAt = { x, y, z };
          const position = { x, y, z: z + FLY_TO_OFFSET_Z };
          const fg = fgRef.current;
          if (fg?.cameraPosition) {
            fg.cameraPosition(position, lookAt, FLY_TO_DURATION_MS);
          }
          navHistory.push({ nodeId, position, lookAt });
          setBreadcrumbTick((t) => t + 1);
        }
      }
    },
    [expandNode, navHistory],
  );
  const handleNodeHover = useCallback(
    (node: { id?: string | number } | null) => hoverNode(node?.id != null ? String(node.id) : null),
    [hoverNode],
  );

  // ── LOD-aware node visibility callback ──
  // Hides nodes entirely when they're outside the visibility budget.
  // This is the biggest performance win — hidden nodes skip all rendering.
  const nodeVisibility = useCallback(
    (node: Record<string, unknown>) => {
      const lod = lodRef.current;
      if (!lod) return true;
      return lod.isVisible(node.id as string);
    },
    [],
  );

  // ── LOD-aware link visibility callback ──
  const linkVisibility = useCallback(
    () => {
      const lod = lodRef.current;
      if (!lod) return true;
      return lod.config.showEdges;
    },
    [],
  );

  // Tier-aware nodeThreeObject callback with caching
  const showHeatmapRef = useRef(showHeatmap);
  showHeatmapRef.current = showHeatmap;

  // Invalidate cache when heatmap toggle changes
  useEffect(() => {
    nodeObjectCacheRef.current.clear();
  }, [showHeatmap]);

  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => {
      const id = node.id as string;
      const activation = (node.activationCurrent as number) ?? 0;
      const entityType = (node.entityType as string) ?? "Other";
      const accessCount = (node.accessCount as number) ?? 0;

      const classifier = classifierRef.current;
      const focusRenderer = focusRendererRef.current;
      const lod = lodRef.current;

      if (!classifier || !focusRenderer) {
        return createFallbackNode(node, showHeatmapRef.current);
      }

      // LOD-aware classification: cap detail based on zoom level
      const detailLevel = lod?.getNodeDetailLevel(id, activation) ?? 2;
      const tier = classifier.classify(id, activation, detailLevel);
      nodeTierMapRef.current.set(id, tier);

      // Return cached object if tier hasn't changed
      const cached = nodeObjectCacheRef.current.get(id);
      if (cached && cached.tier === tier) {
        return cached.obj;
      }

      // Tier changed or new node — create fresh object
      let obj: THREE.Group;

      if (tier === "focus") {
        obj = focusRenderer.createNodeObject(
          id,
          activation,
          entityType,
          accessCount,
          showHeatmapRef.current,
        );
      } else if (tier === "active") {
        // Active tier — batch rendered via ActiveTierRenderer (InstancedMesh)
        activeRendererRef.current?.addNode(id, entityType);
        obj = new THREE.Group();
        obj.userData.tier = "active";
        obj.userData.nodeId = id;
        obj.userData.batchRendered = true;
      } else {
        // Dormant tier — batch rendered via DormantTierRenderer (Points cloud)
        dormantRendererRef.current?.addNode(id, entityType);
        obj = new THREE.Group();
        obj.userData.tier = "dormant";
        obj.userData.nodeId = id;
        obj.userData.batchRendered = true;
      }

      nodeObjectCacheRef.current.set(id, { tier, obj });
      return obj;
    },
    // showHeatmap toggle triggers full rebuild via cache clear above
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showHeatmap],
  );

  // ── Search-to-navigate: fly to selected node when it changes from external sources ──
  const selectedNodeId = useEngramStore((s) => s.selectedNodeId);

  useEffect(() => {
    // Only fly when selection changes from outside (search bar, etc.)
    // Skip if the click handler already flew (it sets prevSelectedRef)
    if (selectedNodeId && selectedNodeId !== prevSelectedRef.current) {
      // Delay to let the node settle into position if just loaded
      const timer = setTimeout(() => {
        const result = flyToNode(selectedNodeId);
        if (result) {
          navHistory.push({ nodeId: selectedNodeId, ...result });
          setBreadcrumbTick((t) => t + 1);
        }
      }, 300);
      prevSelectedRef.current = selectedNodeId;
      return () => clearTimeout(timer);
    }
    prevSelectedRef.current = selectedNodeId;
  }, [selectedNodeId, flyToNode, navHistory]);

  // Breadcrumb trail data
  const breadcrumbData = useMemo(() => {
    // breadcrumbTick forces recompute
    void breadcrumbTick;
    return navHistory.getVisibleTrail(8);
  }, [navHistory, breadcrumbTick]);

  const handleBreadcrumbNavigate = useCallback(
    (index: number) => {
      const nodeId = navHistory.goToIndex(index, flyToBookmark);
      if (nodeId) {
        prevSelectedRef.current = nodeId;
        selectNode(nodeId);
      }
      setBreadcrumbTick((t) => t + 1);
    },
    [navHistory, flyToBookmark, selectNode],
  );

  if (!isLoading && graphData.nodes.length === 0) return <EmptyState />;

  const is3D = renderMode === "3d";

  return (
    <div ref={containerRef} className="relative h-full w-full">
      {import.meta.env.DEV && showStressTest && (
        <Suspense fallback={null}>
          <StressTestPanel />
        </Suspense>
      )}
      {isLoading && (
        <div
          className="absolute inset-0 z-10 flex items-center justify-center"
          style={{ background: "rgba(3, 4, 8, 0.6)" }}
        >
          <div
            className="animate-pulse-soft"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 18,
              color: "var(--accent)",
              fontStyle: "italic",
            }}
          >
            Loading memories...
          </div>
        </div>
      )}
      {is3D ? (
        <ForceGraph3D
          ref={fgRef as React.MutableRefObject<never>}
          graphData={graphData}
          nodeId="id"
          nodeLabel="name"
          nodeThreeObject={nodeThreeObject}
          nodeVisibility={nodeVisibility}
          linkVisibility={linkVisibility}
          linkSource="source"
          linkTarget="target"
          linkCurvature={0}
          linkCurveRotation={0}
          linkResolution={6}
          linkWidth={() => 0}
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(100, 160, 255, 0.5)"}
          linkDirectionalParticles={0}
          linkDirectionalParticleWidth={2.0}
          linkDirectionalParticleColor={() => "#c4b5fd"}
          linkDirectionalParticleSpeed={0.002}
          linkOpacity={0}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onNodeHover={handleNodeHover}
          cooldownTicks={200}
          d3AlphaDecay={0.03}
          backgroundColor="#030408"
          showNavInfo={false}
        />
      ) : (
        <ForceGraph2D
          graphData={graphData}
          nodeId="id"
          nodeLabel="name"
          nodeColor={(n: Record<string, unknown>) =>
            showHeatmap
              ? activationColor((n.activationCurrent as number) ?? 0)
              : entityColor((n.entityType as string) ?? "Other")
          }
          nodeVal={(n: Record<string, unknown>) =>
            2 + ((n.activationCurrent as number) ?? 0) * 12
          }
          linkSource="source"
          linkTarget="target"
          linkWidth={(link: Record<string, unknown>) =>
            Math.max(0.5, ((link.weight as number) ?? 1) * 1.5)
          }
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(100, 160, 255, 0.35)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => "#c4b5fd"}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onNodeHover={handleNodeHover}
          cooldownTicks={200}
          d3AlphaDecay={0.03}
          backgroundColor="#030408"
        />
      )}
      <NodeTooltip />
      <ZoomIndicator lodRef={lodRef} nodeCount={graphData.nodes.length} />
      <NavigationBreadcrumbs
        items={breadcrumbData.items}
        currentIndex={breadcrumbData.currentIndex}
        truncatedLeft={breadcrumbData.truncatedLeft}
        truncatedRight={breadcrumbData.truncatedRight}
        onNavigate={handleBreadcrumbNavigate}
        nodeNames={nodeNames}
      />
      {showHelp && <KeyboardHelpOverlay onClose={() => setShowHelp(false)} />}
      {import.meta.env.DEV && (
        <button
          onClick={() => setShowStressTest((v) => !v)}
          className={showStressTest ? "pill pill-active" : "pill"}
          style={{
            position: "absolute",
            bottom: 12,
            right: 12,
            zIndex: 50,
            fontSize: 11,
          }}
        >
          Stress
        </button>
      )}
    </div>
  );
}

/** Zoom tier indicator overlay */
function ZoomIndicator({
  lodRef,
  nodeCount,
}: {
  lodRef: React.RefObject<LODController | null>;
  nodeCount: number;
}) {
  const [info, setInfo] = useState({ tier: "neighborhood" as string, visible: 0, total: 0 });

  useEffect(() => {
    const interval = setInterval(() => {
      const lod = lodRef.current;
      if (!lod) return;
      const cfg = lod.config;
      const visibleBudget = Math.min(cfg.visibilityBudget, nodeCount);
      setInfo((prev) => {
        if (prev.tier === lod.tier && prev.total === nodeCount && prev.visible === visibleBudget) return prev;
        return { tier: lod.tier, visible: visibleBudget, total: nodeCount };
      });
    }, 500);
    return () => clearInterval(interval);
  }, [lodRef, nodeCount]);

  if (info.total === 0) return null;

  const tierLabels: Record<string, string> = {
    macro: "Macro",
    region: "Region",
    neighborhood: "Neighborhood",
    detail: "Detail",
    synapse: "Synapse",
  };

  return (
    <div
      style={{
        position: "absolute",
        bottom: 12,
        left: 12,
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "4px 10px",
        borderRadius: 6,
        background: "rgba(3, 4, 8, 0.7)",
        backdropFilter: "blur(4px)",
        fontSize: 11,
        color: "rgba(148, 163, 184, 0.8)",
        fontFamily: "var(--font-mono)",
        pointerEvents: "none",
      }}
    >
      <span style={{ color: "var(--accent)", fontWeight: 600 }}>
        {tierLabels[info.tier] ?? info.tier}
      </span>
      <span>{info.total.toLocaleString()} nodes</span>
      {info.visible < info.total && (
        <span style={{ opacity: 0.6 }}>
          ({info.visible.toLocaleString()} visible)
        </span>
      )}
    </div>
  );
}

/**
 * Fallback: create a simple node object when tier system isn't initialized yet.
 * Matches the original createNodeObject signature.
 */
function createFallbackNode(node: Record<string, unknown>, showHeatmap: boolean): THREE.Group {
  const activation = (node.activationCurrent as number) ?? 0;
  const type = (node.entityType as string) ?? "Other";
  const colorHex = showHeatmap ? activationColor(activation) : entityColor(type);
  const color = new THREE.Color(colorHex);
  const coreRadius = 2 + activation * 6;

  const somaGeometry = new THREE.IcosahedronGeometry(coreRadius, 1);
  const somaMaterial = new THREE.MeshBasicMaterial({ color });
  const somaMesh = new THREE.Mesh(somaGeometry, somaMaterial);

  const group = new THREE.Group();
  group.add(somaMesh);

  // Store references for compatibility with animation loop
  group.userData.soma = somaMesh;
  group.userData.tier = "focus";
  group.userData.baseColor = color.getHex();

  return group;
}
