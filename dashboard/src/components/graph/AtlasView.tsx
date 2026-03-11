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
import type { AtlasRegion } from "../../store/types";

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

interface AtlasGraphNode {
  id: string;
  name: string;
  fx: number;
  fy: number;
  fz: number;
  memberCount: number;
  activationScore: number;
  kind: AtlasRegion["kind"];
  dominantEntityTypes: Record<string, number>;
  __region: AtlasRegion;
  [key: string]: unknown;
}

interface AtlasGraphLink {
  source: string;
  target: string;
  weight: number;
  relationshipCount: number;
  [key: string]: unknown;
}

/* ── Helpers ─────────────────────────────────────────── */

function regionColor(region: AtlasRegion, showHeatmap: boolean): string {
  if (showHeatmap) {
    return activationColor(region.activationScore);
  }
  const dominantType = Object.entries(region.dominantEntityTypes).sort(
    (a, b) => b[1] - a[1],
  )[0]?.[0];
  return entityColor(dominantType ?? "Other");
}

function isNewSinceVisit(
  latestEntityCreatedAt: string | null,
  lastAtlasVisitAt: string | null,
): boolean {
  if (!latestEntityCreatedAt || !lastAtlasVisitAt) return false;
  return latestEntityCreatedAt > lastAtlasVisitAt;
}

const POSITION_SCALE = 200;

/* ── Main Component ──────────────────────────────────── */

export function AtlasView() {
  const atlas = useEngramStore((s) => s.atlas);
  const isLoading = useEngramStore((s) => s.isLoading);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const loadRegion = useEngramStore((s) => s.loadRegion);
  const atlasSnapshotId = useEngramStore((s) => s.atlasSnapshotId);
  const lastAtlasVisitAt = useEngramStore((s) => s.lastAtlasVisitAt);
  const recordAtlasVisit = useEngramStore((s) => s.recordAtlasVisit);

  const regions = useDeferredValue(atlas?.regions ?? []);
  const bridges = useDeferredValue(atlas?.bridges ?? []);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const fgRef = useRef<FgRef>(null);
  const sharedRef = useRef<SharedResources | null>(null);
  const registryRef = useRef<DisposalRegistry | null>(null);

  const atlasSnapshotRef = useRef<{
    generatedAt: string | null;
    snapshotId: string | null;
    shouldRecordVisit: boolean;
  }>({
    generatedAt: null,
    snapshotId: null,
    shouldRecordVisit: false,
  });
  const isHistoricalSnapshot = atlasSnapshotId !== null;

  const maxMembers = useMemo(
    () => Math.max(...regions.map((r) => r.memberCount), 1),
    [regions],
  );

  const regionById = useMemo(
    () => new Map(regions.map((r) => [r.id, r])),
    [regions],
  );

  // Only show bridges above median weight
  const visibleBridges = useMemo(() => {
    if (bridges.length === 0) return [];
    const weights = bridges.map((b) => b.weight).sort((a, b) => a - b);
    const median = weights[Math.floor(weights.length * 0.5)] ?? 0;
    return bridges
      .filter((b) => b.weight > median)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 60);
  }, [bridges]);

  // Hovered region's connected bridge targets
  const hoveredConnections = useMemo(() => {
    if (!hoveredId) return new Set<string>();
    const connected = new Set<string>();
    for (const b of bridges) {
      if (b.source === hoveredId) connected.add(b.target);
      if (b.target === hoveredId) connected.add(b.source);
    }
    return connected;
  }, [hoveredId, bridges]);

  const atlasHealth = useMemo(() => {
    const activeRegionCount = regions.filter(
      (r) => r.activationScore >= 0.25,
    ).length;
    const growingRegionCount = regions.filter((r) => r.growth7d > 0).length;
    const newRegionCount = isHistoricalSnapshot
      ? 0
      : regions.filter((r) =>
          isNewSinceVisit(r.latestEntityCreatedAt, lastAtlasVisitAt),
        ).length;
    const recentMemoryCount = regions.reduce(
      (sum, r) => sum + r.growth7d,
      0,
    );
    return { activeRegionCount, growingRegionCount, newRegionCount, recentMemoryCount };
  }, [isHistoricalSnapshot, lastAtlasVisitAt, regions]);

  const hottestRegion = atlas?.stats.hottestRegionId
    ? regionById.get(atlas.stats.hottestRegionId) ?? null
    : null;
  const fastestGrowingRegion = atlas?.stats.fastestGrowingRegionId
    ? regionById.get(atlas.stats.fastestGrowingRegionId) ?? null
    : null;

  // Convert regions + bridges to ForceGraph3D data
  const graphData = useMemo(() => {
    const nodes: AtlasGraphNode[] = regions.map((r) => ({
      id: r.id,
      name: r.label,
      fx: r.x * POSITION_SCALE,
      fy: r.y * POSITION_SCALE,
      fz: r.z * POSITION_SCALE,
      memberCount: r.memberCount,
      activationScore: r.activationScore,
      kind: r.kind,
      dominantEntityTypes: r.dominantEntityTypes,
      __region: r,
    }));

    const links: AtlasGraphLink[] = visibleBridges.map((b) => ({
      source: b.source,
      target: b.target,
      weight: b.weight,
      relationshipCount: b.relationshipCount,
    }));

    return { nodes, links };
  }, [regions, visibleBridges]);

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

  // Bloom + ambient particles
  useEffect(() => {
    const bloomCleanup = setupBloomPass(fgRef as React.RefObject<FgRef>);
    const particles = createAmbientParticles(
      fgRef as React.RefObject<FgRef>,
      150,
      350,
    );

    return () => {
      bloomCleanup();
      particles.cleanup();
    };
  }, []);

  // Set initial camera position
  useEffect(() => {
    const trySetCamera = () => {
      const fg = fgRef.current;
      if (!fg?.cameraPosition) return false;
      fg.cameraPosition(
        { x: 0, y: 60, z: 450 },
        { x: 0, y: 0, z: 18 },
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

  // Animation loop: breathing + particle drift + hover dimming
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

      // Animate breathing + hover dimming on node objects
      for (const child of mainGroup.children) {
        const data = (child as unknown as Record<string, unknown>).__data as
          | Record<string, unknown>
          | undefined;
        if (!data || typeof data.id !== "string") continue;
        if ("source" in data && "target" in data && !("__region" in data)) continue;

        const nodeId = data.id as string;
        const group = child as THREE.Group;

        // Breathing on glow sprite
        const sprite = group.userData.sprite as THREE.Sprite | undefined;
        if (sprite) {
          const baseScale = group.userData.baseGlowScale as number;
          const idx = group.userData.breathIdx as number ?? 0;
          const breath = 1 + 0.06 * Math.sin(time * 0.0008 + idx * 0.9);
          sprite.scale.setScalar(baseScale * breath);

          // Hover dimming
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

        // Dim soma for non-connected
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

  // Atlas visit tracking
  useEffect(() => {
    atlasSnapshotRef.current = {
      generatedAt: atlas?.generatedAt ?? null,
      snapshotId: atlas?.representation.snapshotId ?? null,
      shouldRecordVisit: !isHistoricalSnapshot,
    };
  }, [atlas?.generatedAt, atlas?.representation.snapshotId, isHistoricalSnapshot]);

  useEffect(() => {
    return () => {
      const snapshot = atlasSnapshotRef.current;
      if (!snapshot.shouldRecordVisit) return;
      recordAtlasVisit({
        generatedAt: snapshot.generatedAt,
        snapshotId: snapshot.snapshotId,
      });
    };
  }, [recordAtlasVisit]);

  const showHeatmapRef = useRef(showHeatmap);
  showHeatmapRef.current = showHeatmap;

  // Node rendering callback
  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => {
      const region = node.__region as AtlasRegion;
      if (!region) {
        const g = new THREE.Group();
        g.userData.tier = "atlas";
        return g;
      }

      const shared = sharedRef.current;
      const registry = registryRef.current;

      const colorHex = regionColor(region, showHeatmapRef.current);
      const color = new THREE.Color(colorHex);

      const ratio = region.memberCount / maxMembers;
      const somaRadius = 3 + Math.log2(1 + ratio * 15) * 2;

      // Soma
      const somaMaterial = new THREE.MeshBasicMaterial({ color });
      const somaGeo = shared?.somaGeometryHigh ?? new THREE.IcosahedronGeometry(1, 1);
      const somaMesh = new THREE.Mesh(somaGeo, somaMaterial);
      const scale = region.kind === "identity" ? somaRadius * 1.2 : somaRadius;
      somaMesh.scale.setScalar(scale);

      const group = new THREE.Group();
      group.add(somaMesh);

      if (registry) {
        registry.trackMaterial(region.id, somaMaterial);
      }

      // Dendrites
      const dendriteCount = 4 + Math.min(Math.floor(region.memberCount / 50), 8);
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
          registry.trackGeometry(region.id, geometry);
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
      const glowAlpha = 0.06 + region.activationScore * 0.14;
      const glowSize = somaRadius * (3 + region.activationScore * 2);
      const identityBoost = region.kind === "identity" ? 1.2 : 1;

      const spriteMaterial = shared?.createGlowSpriteMaterial(color, glowAlpha) ??
        new THREE.SpriteMaterial({
          color,
          transparent: true,
          opacity: glowAlpha,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });
      if (registry) {
        registry.trackMaterial(region.id, spriteMaterial);
      }

      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.setScalar(glowSize * identityBoost);
      group.add(sprite);

      // Store refs for animation
      group.userData.soma = somaMesh;
      group.userData.sprite = sprite;
      group.userData.spriteMaterial = spriteMaterial;
      group.userData.baseGlowScale = glowSize * identityBoost;
      group.userData.baseColor = color.getHex();
      group.userData.tier = "atlas";
      group.userData.breathIdx = regions.indexOf(region);

      if (registry) {
        registry.trackGroup(region.id, group);
      }

      return group;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showHeatmap, maxMembers, regions],
  );

  const handleNodeClick = useCallback(
    (node: { id?: string | number }) => {
      const regionId = node.id != null ? String(node.id) : null;
      if (!regionId) return;
      startTransition(() => {
        void loadRegion(regionId, { snapshotId: atlasSnapshotId });
      });
    },
    [loadRegion, atlasSnapshotId],
  );

  const handleNodeHover = useCallback(
    (node: { id?: string | number } | null) => {
      setHoveredId(node?.id != null ? String(node.id) : null);
    },
    [],
  );

  if (!atlas) {
    return (
      <div className="relative h-full w-full" style={{ background: "#030408" }}>
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ color: "var(--text-secondary)", fontSize: 13 }}
        >
          Preparing atlas...
        </div>
      </div>
    );
  }

  if (!isLoading && regions.length === 0) {
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
          style={{ background: "rgba(3, 4, 8, 0.45)" }}
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
            Mapping brain atlas...
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

      {/* Info card -- top left (offset past sidebar) */}
      <div
        className="card"
        style={{
          position: "absolute",
          left: 272,
          top: 12,
          zIndex: 20,
          maxWidth: 320,
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
          Brain Atlas
        </div>
        <div className="display" style={{ fontSize: 20, color: "#fff", lineHeight: 1.1 }}>
          {atlas?.stats.totalRegions.toLocaleString() ?? 0} regions{" "}
          <span style={{ color: "var(--text-muted)", fontSize: 14, fontWeight: 400 }}>
            / {atlas?.stats.totalEntities.toLocaleString() ?? 0} memories
          </span>
        </div>
        <div
          style={{
            marginTop: 8,
            display: "flex",
            gap: 6,
            flexWrap: "wrap",
            color: "var(--text-secondary)",
            fontSize: 11,
          }}
        >
          <span className="pill">
            {visibleBridges.length} / {bridges.length} bridges
          </span>
          {isHistoricalSnapshot && <span className="pill">snapshot</span>}
        </div>
      </div>

      {/* Health card -- top right */}
      <div
        className="card"
        style={{
          position: "absolute",
          right: 12,
          top: 12,
          zIndex: 20,
          width: 260,
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
          Brain Health
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 6,
          }}
        >
          <HealthStat label="Active" value={atlasHealth.activeRegionCount} />
          <HealthStat label="Growing" value={atlasHealth.growingRegionCount} />
          <HealthStat label="New" value={atlasHealth.newRegionCount} />
          <HealthStat label="Recent" value={atlasHealth.recentMemoryCount} />
        </div>
        {(hottestRegion || fastestGrowingRegion) && (
          <div
            style={{
              marginTop: 8,
              display: "grid",
              gap: 4,
              color: "var(--text-secondary)",
              fontSize: 11,
            }}
          >
            {hottestRegion && (
              <div>
                Hotspot: <span style={{ color: "#fff" }}>{hottestRegion.label}</span>
              </div>
            )}
            {fastestGrowingRegion && (
              <div>
                Growing: <span style={{ color: "#fff" }}>{fastestGrowingRegion.label}</span>
              </div>
            )}
          </div>
        )}
        <div
          style={{
            marginTop: 10,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <div
            style={{
              flex: 1,
              height: 6,
              borderRadius: 999,
              background:
                "linear-gradient(90deg, rgba(56,189,248,0.25), rgba(34,211,238,0.6), rgba(250,204,21,0.75))",
              border: "1px solid rgba(255,255,255,0.06)",
            }}
          />
          <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)" }}>
            {showHeatmap ? "heat" : "type"}
          </span>
        </div>
      </div>

      {/* Entity type legend */}
      <EntityTypeLegend />

      {/* Hint */}
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
        click a region to explore
      </div>
    </div>
  );
}

const LEGEND_TYPES = [
  "Person",
  "Organization",
  "Project",
  "Technology",
  "Concept",
  "Location",
  "Event",
  "Identifier",
] as const;

function EntityTypeLegend() {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 40,
        left: 272,
        zIndex: 20,
        padding: "8px 12px",
        borderRadius: 8,
        background: "rgba(3, 4, 8, 0.6)",
        backdropFilter: "blur(4px)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
        display: "flex",
        flexDirection: "column",
        gap: 3,
        pointerEvents: "none",
      }}
    >
      {LEGEND_TYPES.map((type) => (
        <div
          key={type}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: entityColor(type),
              flexShrink: 0,
              boxShadow: `0 0 4px ${entityColor(type)}40`,
            }}
          />
          <span
            style={{
              fontSize: 10,
              color: "rgba(148, 163, 184, 0.7)",
              fontFamily: "var(--font-body)",
            }}
          >
            {type}
          </span>
        </div>
      ))}
    </div>
  );
}

function HealthStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="pill" style={{ justifyContent: "space-between" }}>
      <span>{label}</span>
      <span className="mono tabular-nums">{value}</span>
    </div>
  );
}
