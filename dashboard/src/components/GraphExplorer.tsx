import { useCallback, useRef, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import ForceGraph2D from "react-force-graph-2d";
import * as THREE from "three";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { useEngramStore } from "../store";
import { activationColor, entityColor } from "../lib/colors";
import { NodeTooltip } from "./NodeTooltip";
import { EmptyState } from "./EmptyState";

// Tier system
import { TierClassifier, type Tier } from "./graph/TierClassifier";
import { FocusTierRenderer } from "./graph/FocusTierRenderer";
import { ActiveTierRenderer } from "./graph/ActiveTierRenderer";
import { DormantTierRenderer } from "./graph/DormantTierRenderer";
import { SharedResources } from "./graph/SharedResources";
import { AdjacencyMap } from "./graph/AdjacencyMap";
import { AnimationBudget } from "./graph/AnimationBudget";
import { DisposalRegistry } from "./graph/DisposalRegistry";
import { useStableGraphData, useActivationRef, useNodeDataRef } from "../store/graphSelectors";

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
  graphData?: () => {
    nodes: Array<Record<string, unknown>>;
    links: Array<Record<string, unknown>>;
  };
} | null;

export function GraphExplorer() {
  const isLoading = useEngramStore((s) => s.isLoading);
  const renderMode = useEngramStore((s) => s.renderMode);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const showEdgeLabels = useEngramStore((s) => s.showEdgeLabels);
  const selectNode = useEngramStore((s) => s.selectNode);
  const hoverNode = useEngramStore((s) => s.hoverNode);
  const expandNode = useEngramStore((s) => s.expandNode);

  const fgRef = useRef<FgRef>(null);
  const graphData = useStableGraphData();
  const activationRef = useActivationRef();
  const nodeDataRef = useNodeDataRef();

  // ── Tier system refs (stable across renders) ──
  const sharedRef = useRef<SharedResources | null>(null);
  const registryRef = useRef<DisposalRegistry | null>(null);
  const classifierRef = useRef<TierClassifier | null>(null);
  const focusRendererRef = useRef<FocusTierRenderer | null>(null);
  const activeRendererRef = useRef<ActiveTierRenderer | null>(null);
  const dormantRendererRef = useRef<DormantTierRenderer | null>(null);
  const adjacencyRef = useRef<AdjacencyMap | null>(null);
  const budgetRef = useRef<AnimationBudget | null>(null);

  // Track which nodes are in which tier for batch renderers
  const nodeTierMapRef = useRef<Map<string, Tier>>(new Map());

  // Initialize tier system
  useEffect(() => {
    const shared = new SharedResources();
    const registry = new DisposalRegistry();
    sharedRef.current = shared;
    registryRef.current = registry;
    classifierRef.current = new TierClassifier();
    focusRendererRef.current = new FocusTierRenderer(shared, registry);
    activeRendererRef.current = new ActiveTierRenderer(shared);
    dormantRendererRef.current = new DormantTierRenderer();
    adjacencyRef.current = new AdjacencyMap();
    budgetRef.current = new AnimationBudget();

    return () => {
      registry.disposeAll();
      shared.dispose();
      activeRendererRef.current?.dispose();
      dormantRendererRef.current?.dispose();
      classifierRef.current?.clear();
      adjacencyRef.current?.clear();
      budgetRef.current?.reset();
    };
  }, []);

  // Rebuild adjacency map when graph structure changes
  useEffect(() => {
    if (adjacencyRef.current && graphData.links) {
      adjacencyRef.current.rebuild(graphData.links);
    }
  }, [graphData]);

  // Batch renderers kept for future large-graph optimization but not scene-added.
  // All tiers now return visible meshes from nodeThreeObject directly.

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
      if (charge?.strength) charge.strength(-40);
      if (charge?.distanceMax) charge.distanceMax(300);

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
  const bloomAddedRef = useRef(false);

  useEffect(() => {
    if (renderMode !== "3d") {
      bloomAddedRef.current = false;
      return;
    }

    const addBloom = () => {
      const fg = fgRef.current as {
        renderer?: () => THREE.WebGLRenderer;
        postProcessingComposer?: () => {
          passes: unknown[];
          addPass: (pass: unknown) => void;
        };
      } | null;
      if (!fg?.renderer) return false;
      if (bloomAddedRef.current) return true;

      try {
        const renderer = fg.renderer();
        const bloomPass = new UnrealBloomPass(
          new THREE.Vector2(
            renderer.domElement.width,
            renderer.domElement.height,
          ),
          0.5,
          0.4,
          0.6,
        );

        if (fg.postProcessingComposer) {
          fg.postProcessingComposer().addPass(bloomPass);
        }

        bloomAddedRef.current = true;
      } catch {
        return false;
      }
      return true;
    };

    if (!addBloom()) {
      const id = setInterval(() => {
        if (addBloom()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }
  }, [renderMode]);

  // ── Ambient neurotransmitter particles ──
  const particlesRef = useRef<THREE.Points | null>(null);

  useEffect(() => {
    if (renderMode !== "3d") return;

    const addParticles = () => {
      const fg = fgRef.current as { scene?: () => THREE.Scene } | null;
      if (!fg?.scene) return false;
      if (particlesRef.current) return true;

      const scene = fg.scene();
      const count = 200;
      const positions = new Float32Array(count * 3);

      for (let i = 0; i < count; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 50 + Math.random() * 450;
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );

      const material = new THREE.PointsMaterial({
        size: 0.5,
        color: 0x67e8f9,
        transparent: true,
        opacity: 0.2,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      });

      const particles = new THREE.Points(geometry, material);
      particles.userData.driftAxes = Array.from({ length: count }, () =>
        new THREE.Vector3(
          Math.random() - 0.5,
          Math.random() - 0.5,
          Math.random() - 0.5,
        ).normalize(),
      );
      scene.add(particles);
      particlesRef.current = particles;
      return true;
    };

    if (!addParticles()) {
      const id = setInterval(() => {
        if (addParticles()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }

    return () => {
      if (particlesRef.current) {
        const fg = fgRef.current as { scene?: () => THREE.Scene } | null;
        if (fg?.scene) {
          fg.scene().remove(particlesRef.current);
        }
        particlesRef.current.geometry.dispose();
        (particlesRef.current.material as THREE.Material).dispose();
        particlesRef.current = null;
      }
    };
  }, [renderMode]);

  // ── Optimized animation loop with tier-aware rendering ──
  const dirtyNodesRef = useRef<Set<string>>(new Set());
  const emittedPulsesRef = useRef<Set<string>>(new Set());
  const cascadeQueueRef = useRef<Map<string, CascadeEntry>>(new Map());
  const cascadedPulsesRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (renderMode !== "3d") return;

    let rafId: number;
    const startTime = Date.now();

    const animate = () => {
      rafId = requestAnimationFrame(animate);

      const pulses = useEngramStore.getState().activationPulses;
      const now = Date.now();
      const time = now - startTime;
      const fg = fgRef.current as FgRef;

      // 1. Particle drift (unchanged — already efficient)
      if (particlesRef.current) {
        const positions = particlesRef.current.geometry.attributes.position;
        const posArray = positions.array as Float32Array;
        const axes = particlesRef.current.userData.driftAxes as THREE.Vector3[];
        const rotSpeed = 0.0001;

        for (let i = 0; i < axes.length; i++) {
          const x = posArray[i * 3];
          const y = posArray[i * 3 + 1];
          const z = posArray[i * 3 + 2];
          const pos = new THREE.Vector3(x, y, z);
          pos.applyAxisAngle(axes[i], rotSpeed);
          posArray[i * 3] = pos.x;
          posArray[i * 3 + 1] = pos.y;
          posArray[i * 3 + 2] = pos.z;
        }
        positions.needsUpdate = true;
      }

      if (!fg?.graphData) return;
      const gd = fg.graphData();

      const classifier = classifierRef.current;
      const focusRenderer = focusRendererRef.current;
      const adjacency = adjacencyRef.current;
      const budget = budgetRef.current;

      if (!classifier || !focusRenderer || !adjacency || !budget) return;

      // 2. Classify all nodes and manage tier transitions
      classifier.updateTransitions(now);

      const activations = activationRef.current;
      const nodePositionMap = new Map<string, { x?: number; y?: number; z?: number }>();
      const activePulseIds = new Set<string>();
      const focusNodeIds: string[] = [];

      // Build position map + reclassify nodes
      for (const node of gd.nodes) {
        const id = node.id as string;
        const activation = activations.get(id) ?? 0;
        nodePositionMap.set(id, {
          x: node.x as number | undefined,
          y: node.y as number | undefined,
          z: node.z as number | undefined,
        });

        const newTier = classifier.classify(id, activation);
        const oldTier = nodeTierMapRef.current.get(id);

        // Track tier transitions (nodeThreeObject handles visual creation)
        if (oldTier !== newTier) {
          nodeTierMapRef.current.set(id, newTier);
        }

        if (newTier === "focus") focusNodeIds.push(id);
      }

      // Clean up nodes that were removed from graph
      for (const [id] of nodeTierMapRef.current) {
        if (!activations.has(id) && !gd.nodes.some((n: Record<string, unknown>) => n.id === id)) {
          registryRef.current?.disposeNode(id);
          classifier.removeNode(id);
          nodeTierMapRef.current.delete(id);
        }
      }

      // 4. Fire queued cascades
      for (const [key, entry] of cascadeQueueRef.current) {
        if (now >= entry.fireAt) {
          cascadeQueueRef.current.delete(key);
          useEngramStore.getState().addActivationPulse({
            entityId: entry.targetId,
            name: "",
            entityType: "",
            activation: entry.intensity,
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
          const graphNode = gd.nodes.find(
            (n: Record<string, unknown>) => n.id === pulse.entityId,
          );
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
          const graphNode = gd.nodes.find(
            (n: Record<string, unknown>) => n.id === nodeId,
          );
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

        const graphNode = gd.nodes.find(
          (n: Record<string, unknown>) => n.id === id,
        );
        if (!graphNode) continue;

        const threeObj = graphNode.__threeObj as THREE.Group | undefined;
        if (threeObj && threeObj.userData.tier === "focus") {
          focusRenderer.animateBreathing(threeObj, time, i);
        }
      }

      // Housekeeping
      if (emittedPulsesRef.current.size > 200)
        emittedPulsesRef.current.clear();
      if (cascadedPulsesRef.current.size > 200)
        cascadedPulsesRef.current.clear();
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [renderMode, activationRef, nodeDataRef]);

  const handleNodeClick = useCallback(
    (node: { id?: string | number }) => selectNode(node.id != null ? String(node.id) : null),
    [selectNode],
  );
  const handleNodeRightClick = useCallback(
    (node: { id?: string | number }) => {
      if (node.id != null) expandNode(String(node.id));
    },
    [expandNode],
  );
  const handleNodeHover = useCallback(
    (node: { id?: string | number } | null) => hoverNode(node?.id != null ? String(node.id) : null),
    [hoverNode],
  );

  // Tier-aware nodeThreeObject callback
  const showHeatmapRef = useRef(showHeatmap);
  showHeatmapRef.current = showHeatmap;

  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => {
      const id = node.id as string;
      const activation = (node.activationCurrent as number) ?? 0;
      const entityType = (node.entityType as string) ?? "Other";
      const accessCount = (node.accessCount as number) ?? 0;

      const classifier = classifierRef.current;
      const focusRenderer = focusRendererRef.current;

      if (!classifier || !focusRenderer) {
        // Fallback: always render full neuron if tier system not ready
        return createFallbackNode(node, showHeatmapRef.current);
      }

      const tier = classifier.classify(id, activation);
      nodeTierMapRef.current.set(id, tier);

      if (tier === "focus") {
        // Full neuron mesh from FocusTierRenderer
        return focusRenderer.createNodeObject(
          id,
          activation,
          entityType,
          accessCount,
          showHeatmapRef.current,
        );
      }

      // Active/Dormant: visible simple mesh (no batch renderer indirection)
      const colorHex = showHeatmapRef.current
        ? activationColor(activation)
        : entityColor(entityType);
      const color = new THREE.Color(colorHex);

      if (tier === "active") {
        // Medium sphere with glow sprite
        const radius = 2 + activation * 4;
        const somaMesh = new THREE.Mesh(
          sharedRef.current!.activeSphereGeometry,
          new THREE.MeshBasicMaterial({ color }),
        );
        somaMesh.scale.setScalar(radius);

        const group = new THREE.Group();
        group.add(somaMesh);

        // Small glow sprite for visibility
        const spriteMat = sharedRef.current!.createGlowSpriteMaterial(
          color,
          0.06 + activation * 0.08,
        );
        const sprite = new THREE.Sprite(spriteMat);
        const glowSize = radius * 3;
        sprite.scale.set(glowSize, glowSize, 1);
        group.add(sprite);

        group.userData.soma = somaMesh;
        group.userData.sprite = sprite;
        group.userData.tier = "active";
        group.userData.nodeId = id;
        group.userData.baseColor = color.getHex();
        return group;
      }

      // Dormant: small sphere dot
      const radius = 1.2 + activation * 2;
      const somaMesh = new THREE.Mesh(
        sharedRef.current!.somaGeometryLow,
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.7,
        }),
      );
      somaMesh.scale.setScalar(radius);

      const group = new THREE.Group();
      group.add(somaMesh);

      group.userData.soma = somaMesh;
      group.userData.tier = "dormant";
      group.userData.nodeId = id;
      group.userData.baseColor = color.getHex();
      return group;
    },
    // showHeatmap toggle triggers full rebuild via showHeatmapRef
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showHeatmap],
  );

  if (!isLoading && graphData.nodes.length === 0) return <EmptyState />;

  const is3D = renderMode === "3d";

  return (
    <div className="relative h-full w-full">
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
          linkSource="source"
          linkTarget="target"
          linkCurvature={0.25}
          linkCurveRotation={(link: Record<string, unknown>) =>
            (link.__curveRotation as number) ?? 0
          }
          linkResolution={12}
          linkWidth={(link: Record<string, unknown>) =>
            Math.max(0.4, ((link.weight as number) ?? 1) * 1.2)
          }
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(100, 160, 255, 0.5)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={2.0}
          linkDirectionalParticleColor={() => "#c4b5fd"}
          linkDirectionalParticleSpeed={0.002}
          linkOpacity={0.6}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onNodeHover={handleNodeHover}
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
          backgroundColor="#030408"
        />
      )}
      <NodeTooltip />
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
