import {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { useEngramStore } from "../../store";
import { activationColor, entityColor } from "../../lib/colors";
import { EmptyState } from "../EmptyState";
import { SharedResources } from "./SharedResources";
import { DisposalRegistry } from "./DisposalRegistry";
import { setupBloomPass, createAmbientParticles } from "./NeuralSceneSetup";
import type { RegionGraphNode } from "../../store/types";

/* ── Types ──────────────────────────────────────────── */

type FgRef = {
  d3Force: (name: string) => unknown;
  scene?: () => THREE.Scene;
  renderer?: () => THREE.WebGLRenderer;
  camera?: () => THREE.Camera;
  postProcessingComposer?: () => unknown;
  cameraPosition?: (
    pos: { x: number; y: number; z: number },
    lookAt?: { x: number; y: number; z: number },
    transitionMs?: number,
  ) => void;
} | null;

interface RegionFgNode {
  id: string;
  name: string;
  fx: number;
  fy: number;
  fz: number;
  kind: RegionGraphNode["kind"];
  representedEntityCount: number;
  activationScore: number;
  entityId?: string;
  entityType?: string;
  regionId?: string;
  __node: RegionGraphNode;
  [key: string]: unknown;
}

interface RegionFgLink {
  source: string;
  target: string;
  weight: number;
  [key: string]: unknown;
}

/* ── Helpers ─────────────────────────────────────────── */

function nodeColor(node: RegionGraphNode, showHeatmap: boolean): string {
  if (showHeatmap) return activationColor(node.activationScore);
  if (node.entityType) return entityColor(node.entityType);
  if (node.kind === "bridge") return "#60a5fa";
  if (node.kind === "cluster") return "#22d3ee";
  return "#94a3b8";
}

const POSITION_SCALE = 150;

/* ── Main Component ──────────────────────────────────── */

export function RegionView() {
  const regionData = useEngramStore((s) => s.regionData);
  const isLoading = useEngramStore((s) => s.isLoading);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const atlasSnapshotId = useEngramStore((s) => s.atlasSnapshotId);
  const loadNeighborhood = useEngramStore((s) => s.loadNeighborhood);
  const loadRegion = useEngramStore((s) => s.loadRegion);
  const lastAtlasVisitAt = useEngramStore((s) => s.lastAtlasVisitAt);
  const selectNode = useEngramStore((s) => s.selectNode);

  const nodes = useDeferredValue(regionData?.nodes ?? []);
  const edges = useDeferredValue(regionData?.edges ?? []);
  const isHistoricalSnapshot = atlasSnapshotId !== null;
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const fgRef = useRef<FgRef>(null);
  const sharedRef = useRef<SharedResources | null>(null);
  const registryRef = useRef<DisposalRegistry | null>(null);

  const maxEntities = useMemo(
    () => Math.max(...nodes.map((n) => n.representedEntityCount), 1),
    [nodes],
  );

  // Filter edges
  const visibleEdges = useMemo(() => {
    if (edges.length === 0) return [];
    const weights = edges.map((e) => e.weight).sort((a, b) => a - b);
    const median = weights[Math.floor(weights.length * 0.4)] ?? 0;
    return edges.filter((e) => e.weight > median).slice(0, 80);
  }, [edges]);

  // Hovered connections
  const hoveredConnections = useMemo(() => {
    if (!hoveredId) return new Set<string>();
    const connected = new Set<string>();
    for (const e of edges) {
      if (e.source === hoveredId) connected.add(e.target);
      if (e.target === hoveredId) connected.add(e.source);
    }
    return connected;
  }, [hoveredId, edges]);

  const isNewSinceAtlasVisit =
    !isHistoricalSnapshot &&
    Boolean(
      regionData?.region.latestEntityCreatedAt &&
        lastAtlasVisitAt &&
        regionData.region.latestEntityCreatedAt > lastAtlasVisitAt,
    );

  // Convert to ForceGraph3D data
  const graphData = useMemo(() => {
    const fgNodes: RegionFgNode[] = nodes.map((n) => ({
      id: n.id,
      name: n.label,
      fx: n.x * POSITION_SCALE,
      fy: n.y * POSITION_SCALE,
      fz: n.z * POSITION_SCALE,
      kind: n.kind,
      representedEntityCount: n.representedEntityCount,
      activationScore: n.activationScore,
      entityId: n.entityId,
      entityType: n.entityType,
      regionId: n.regionId,
      __node: n,
    }));

    const fgLinks: RegionFgLink[] = visibleEdges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
    }));

    return { nodes: fgNodes, links: fgLinks };
  }, [nodes, visibleEdges]);

  // Initialize shared resources
  useEffect(() => {
    const shared = new SharedResources();
    const registry = new DisposalRegistry();
    sharedRef.current = shared;
    registryRef.current = registry;

    return () => {
      registry.disposeAll();
      shared.dispose();
    };
  }, []);

  // Bloom + particles
  useEffect(() => {
    const bloomCleanup = setupBloomPass(fgRef as React.RefObject<FgRef>);
    const particles = createAmbientParticles(
      fgRef as React.RefObject<FgRef>,
      120,
      300,
    );

    return () => {
      bloomCleanup();
      particles.cleanup();
    };
  }, []);

  // Set initial camera
  useEffect(() => {
    const trySetCamera = () => {
      const fg = fgRef.current;
      if (!fg?.cameraPosition) return false;
      fg.cameraPosition(
        { x: 0, y: 50, z: 380 },
        { x: 0, y: 0, z: 10 },
        0,
      );
      return true;
    };
    if (!trySetCamera()) {
      const id = setInterval(() => {
        if (trySetCamera()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }
  }, []);

  // Animation loop: breathing + hover dimming
  useEffect(() => {
    let rafId: number;
    const startTime = Date.now();

    const animate = () => {
      rafId = requestAnimationFrame(animate);
      const time = Date.now() - startTime;
      const fg = fgRef.current as FgRef;
      if (!fg?.scene) return;

      const scene = fg.scene();
      const mainGroup = scene.children.find(
        (c: THREE.Object3D) => c.type === "Group" && c.children.length > 2,
      ) as THREE.Group | undefined;
      if (!mainGroup) return;

      for (const child of mainGroup.children) {
        const data = (child as unknown as Record<string, unknown>).__data as
          | Record<string, unknown>
          | undefined;
        if (!data || typeof data.id !== "string") continue;
        if ("source" in data && "target" in data && !("__node" in data)) continue;

        const nodeId = data.id as string;
        const group = child as THREE.Group;

        const sprite = group.userData.sprite as THREE.Sprite | undefined;
        if (sprite) {
          const baseScale = group.userData.baseGlowScale as number;
          const idx = group.userData.breathIdx as number ?? 0;
          const breath = 1 + 0.06 * Math.sin(time * 0.0008 + idx * 0.9);
          sprite.scale.setScalar(baseScale * breath);

          const mat = group.userData.spriteMaterial as THREE.SpriteMaterial | undefined;
          if (mat) {
            const baseOpacity = mat.userData?.baseOpacity ?? mat.opacity;
            if (!mat.userData?.baseOpacity) {
              mat.userData = { baseOpacity };
            }
            if (hoveredId && nodeId !== hoveredId && !hoveredConnections.has(nodeId)) {
              mat.opacity = baseOpacity * 0.15;
            } else {
              mat.opacity = baseOpacity;
            }
          }
        }

        const soma = group.userData.soma as THREE.Mesh | undefined;
        if (soma) {
          const somaMat = soma.material as THREE.MeshBasicMaterial;
          if (hoveredId && nodeId !== hoveredId && !hoveredConnections.has(nodeId)) {
            somaMat.opacity = 0.2;
            somaMat.transparent = true;
          } else {
            somaMat.opacity = 1;
            somaMat.transparent = false;
          }
        }
      }
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [hoveredId, hoveredConnections]);

  const showHeatmapRef = useRef(showHeatmap);
  showHeatmapRef.current = showHeatmap;

  // Node rendering
  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => {
      const regionNode = node.__node as RegionGraphNode;
      if (!regionNode) {
        const g = new THREE.Group();
        g.userData.tier = "region";
        return g;
      }

      const shared = sharedRef.current;
      const registry = registryRef.current;

      const colorHex = nodeColor(regionNode, showHeatmapRef.current);
      const color = new THREE.Color(colorHex);

      // Size based on kind
      let somaRadius: number;
      let dendriteCount: number;
      if (regionNode.kind === "hub") {
        somaRadius = 2 + (regionNode.representedEntityCount / maxEntities) * 3;
        dendriteCount = 2 + Math.min(Math.floor(regionNode.representedEntityCount / 2), 2);
      } else if (regionNode.kind === "cluster") {
        const ratio = regionNode.representedEntityCount / maxEntities;
        somaRadius = 5 + ratio * 4;
        dendriteCount = 4 + Math.min(Math.floor(regionNode.representedEntityCount / 30), 4);
      } else {
        // bridge / schema
        somaRadius = 3 + (regionNode.representedEntityCount / maxEntities) * 3;
        dendriteCount = 3 + Math.min(Math.floor(regionNode.representedEntityCount / 20), 3);
      }

      // Soma
      const somaMaterial = new THREE.MeshBasicMaterial({ color });
      const somaGeo = shared?.somaGeometryHigh ?? new THREE.IcosahedronGeometry(1, 1);
      const somaMesh = new THREE.Mesh(somaGeo, somaMaterial);
      somaMesh.scale.setScalar(somaRadius);

      const group = new THREE.Group();
      group.add(somaMesh);

      if (registry) {
        registry.trackMaterial(regionNode.id, somaMaterial);
      }

      // Dendrites
      for (let i = 0; i < dendriteCount; i++) {
        const dir = new THREE.Vector3(
          Math.random() - 0.5,
          Math.random() - 0.5,
          Math.random() - 0.5,
        ).normalize();

        const length = somaRadius * (1.5 + Math.random() * 1.5);
        const mid = dir
          .clone()
          .multiplyScalar(length * 0.5)
          .add(
            new THREE.Vector3(
              (Math.random() - 0.5) * somaRadius * 0.6,
              (Math.random() - 0.5) * somaRadius * 0.6,
              (Math.random() - 0.5) * somaRadius * 0.6,
            ),
          );
        const tip = dir.clone().multiplyScalar(length);

        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(0, 0, 0),
          mid,
          tip,
        );
        const points = curve.getPoints(6);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const colors = new Float32Array(points.length * 3);
        for (let j = 0; j < points.length; j++) {
          const fade = 1 - j / (points.length - 1);
          colors[j * 3] = color.r * (0.3 + fade * 0.7);
          colors[j * 3 + 1] = color.g * (0.3 + fade * 0.7);
          colors[j * 3 + 2] = color.b * (0.3 + fade * 0.7);
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        if (registry) {
          registry.trackGeometry(regionNode.id, geometry);
        }

        const material = shared?.getDendriteMaterial(i, dendriteCount) ??
          new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.35,
            depthWrite: false,
          });
        group.add(new THREE.Line(geometry, material));
      }

      // Glow sprite
      const glowAlpha = 0.05 + regionNode.activationScore * 0.12;
      const glowSize = somaRadius * (3 + regionNode.activationScore * 2);

      const spriteMaterial = shared?.createGlowSpriteMaterial(color, glowAlpha) ??
        new THREE.SpriteMaterial({
          color,
          transparent: true,
          opacity: glowAlpha,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });
      if (registry) {
        registry.trackMaterial(regionNode.id, spriteMaterial);
      }

      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.setScalar(glowSize);
      group.add(sprite);

      group.userData.soma = somaMesh;
      group.userData.sprite = sprite;
      group.userData.spriteMaterial = spriteMaterial;
      group.userData.baseGlowScale = glowSize;
      group.userData.baseColor = color.getHex();
      group.userData.tier = "region";
      group.userData.breathIdx = nodes.indexOf(regionNode);

      if (registry) {
        registry.trackGroup(regionNode.id, group);
      }

      return group;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showHeatmap, maxEntities, nodes],
  );

  const handleNodeClick = useCallback(
    (node: { id?: string | number } & Record<string, unknown>) => {
      const regionNode = node.__node as RegionGraphNode | undefined;
      if (!regionNode) return;

      if (regionNode.entityId) {
        startTransition(() => {
          selectNode(regionNode.entityId ?? null);
        });
        void loadNeighborhood(regionNode.entityId, 2, {
          regionId: regionData?.region.id ?? null,
        });
        return;
      }
      if (regionNode.regionId) {
        startTransition(() => {
          void loadRegion(regionNode.regionId!, { snapshotId: atlasSnapshotId });
        });
      }
    },
    [selectNode, loadNeighborhood, loadRegion, regionData?.region.id, atlasSnapshotId],
  );

  const handleNodeHover = useCallback(
    (node: { id?: string | number } | null) => {
      setHoveredId(node?.id != null ? String(node.id) : null);
    },
    [],
  );

  if (!regionData) {
    return (
      <div className="relative h-full w-full" style={{ background: "#030408" }}>
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ color: "var(--text-secondary)", fontSize: 13 }}
        >
          Preparing region...
        </div>
      </div>
    );
  }

  if (!isLoading && nodes.length === 0) {
    return <EmptyState />;
  }

  return (
    <div
      className="relative h-full w-full overflow-hidden"
      style={{ background: "#030408" }}
    >
      {isLoading && (
        <div
          className="absolute inset-0 z-10 flex items-center justify-center"
          style={{ background: "rgba(3, 4, 8, 0.4)" }}
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
            Refining region map...
          </div>
        </div>
      )}

      <ForceGraph3D
        ref={fgRef as React.MutableRefObject<never>}
        graphData={graphData}
        nodeId="id"
        nodeLabel="name"
        nodeThreeObject={nodeThreeObject}
        linkSource="source"
        linkTarget="target"
        linkWidth={() => 0}
        linkOpacity={0}
        linkDirectionalParticles={(link: Record<string, unknown>) =>
          1 + Math.floor((link.weight as number) ?? 1)
        }
        linkDirectionalParticleWidth={1.5}
        linkDirectionalParticleColor={() => "#67e8f9"}
        linkDirectionalParticleSpeed={0.003}
        linkCurvature={0.15}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={1}
        d3AlphaDecay={1}
        backgroundColor="#030408"
        showNavInfo={false}
      />

      {/* Info card -- offset past sidebar */}
      <div
        className="card"
        style={{
          position: "absolute",
          left: 272,
          top: 12,
          zIndex: 20,
          maxWidth: 340,
          padding: "12px 14px",
          borderRadius: "var(--radius-md)",
          pointerEvents: "auto",
        }}
      >
        <div
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--accent)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            marginBottom: 6,
          }}
        >
          Region Map
        </div>
        <div className="display" style={{ fontSize: 22, color: "#fff", lineHeight: 1.08 }}>
          {regionData.region.label}
        </div>
        {regionData.region.subtitle && (
          <div
            style={{
              marginTop: 5,
              fontSize: 12,
              lineHeight: 1.4,
              color: "var(--text-secondary)",
            }}
          >
            {regionData.region.subtitle}
          </div>
        )}
        <div
          style={{
            marginTop: 8,
            display: "flex",
            gap: 6,
            flexWrap: "wrap",
            fontSize: 11,
          }}
        >
          <span className="pill">
            {regionData.region.memberCount.toLocaleString()} memories
          </span>
          <span className="pill">
            {(regionData.region.activationScore * 100).toFixed(0)}% heat
          </span>
          {regionData.region.growth7d > 0 && (
            <span className="pill">+{regionData.region.growth7d} this week</span>
          )}
          {isHistoricalSnapshot && <span className="pill">snapshot</span>}
          {isNewSinceAtlasVisit && <span className="pill">new</span>}
        </div>
      </div>

      {/* Top entities card -- top right */}
      <div
        className="card"
        style={{
          position: "absolute",
          right: 12,
          top: 12,
          zIndex: 20,
          width: 220,
          padding: "12px 14px",
          borderRadius: "var(--radius-md)",
          pointerEvents: "auto",
        }}
      >
        <div
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--accent)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            marginBottom: 8,
          }}
        >
          Top Entities
        </div>
        <div style={{ display: "grid", gap: 6 }}>
          {regionData.topEntities.length === 0 && (
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
              No entities yet.
            </div>
          )}
          {regionData.topEntities.map((entity) => {
            const color = entityColor(entity.entityType);
            return (
              <button
                key={entity.id}
                type="button"
                disabled={isHistoricalSnapshot}
                onClick={() => {
                  if (isHistoricalSnapshot) return;
                  startTransition(() => {
                    selectNode(entity.id);
                  });
                  void loadNeighborhood(entity.id, 2, {
                    regionId: regionData.region.id,
                  });
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  width: "100%",
                  padding: "6px 8px",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border)",
                  background: "rgba(5, 10, 18, 0.82)",
                  color: "#fff",
                  textAlign: "left",
                  cursor: isHistoricalSnapshot ? "default" : "pointer",
                  opacity: isHistoricalSnapshot ? 0.72 : 1,
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: color,
                    boxShadow: `0 0 8px ${color}55`,
                    flexShrink: 0,
                  }}
                />
                <span style={{ flex: 1, minWidth: 0 }}>
                  <span
                    className="display"
                    style={{
                      display: "block",
                      fontSize: 13,
                      lineHeight: 1.05,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {entity.name}
                  </span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 9,
                      color: "var(--text-muted)",
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                    }}
                  >
                    {entity.entityType}
                  </span>
                </span>
                <span
                  className="mono tabular-nums"
                  style={{ fontSize: 10, color: activationColor(entity.activationCurrent) }}
                >
                  {(entity.activationCurrent * 100).toFixed(0)}%
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Bottom hint */}
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 20,
          padding: "4px 12px",
          borderRadius: 999,
          background: "rgba(3, 4, 8, 0.6)",
          backdropFilter: "blur(4px)",
          fontSize: 11,
          color: "rgba(148, 163, 184, 0.6)",
          fontFamily: "var(--font-mono)",
          pointerEvents: "none",
        }}
      >
        click to explore entities
      </div>
    </div>
  );
}
