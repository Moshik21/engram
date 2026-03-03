import { useCallback, useMemo, useRef, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import ForceGraph2D from "react-force-graph-2d";
import * as THREE from "three";
import { useEngramStore } from "../store";
import { activationColor, entityColor } from "../lib/colors";
import { NodeTooltip } from "./NodeTooltip";
import { EmptyState } from "./EmptyState";

/**
 * Create a custom THREE.js node object with glow halo.
 */
function createNodeObject(node: Record<string, unknown>, showHeatmap: boolean) {
  const activation = (node.activationCurrent as number) ?? 0;
  const type = (node.entityType as string) ?? "Other";

  // Core color: activation heatmap or type-based
  const colorHex = showHeatmap ? activationColor(activation) : entityColor(type);

  // Core sphere: size scales with activation
  const coreRadius = 2 + activation * 6;
  const coreGeometry = new THREE.SphereGeometry(coreRadius, 16, 12);
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: new THREE.Color(colorHex),
  });
  const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);

  // Outer glow ring (sprite) — intensity scales with activation
  const glowAlpha = 0.08 + activation * 0.25;
  const glowSize = coreRadius * (3 + activation * 2);

  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, `rgba(255, 255, 255, ${glowAlpha})`);
  gradient.addColorStop(0.4, `rgba(255, 255, 255, ${glowAlpha * 0.3})`);
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);

  const texture = new THREE.CanvasTexture(canvas);
  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    color: new THREE.Color(colorHex),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(glowSize, glowSize, 1);

  const group = new THREE.Group();
  group.add(sprite);
  group.add(coreMesh);

  return group;
}

export function GraphExplorer() {
  const nodes = useEngramStore((s) => s.nodes);
  const edges = useEngramStore((s) => s.edges);
  const isLoading = useEngramStore((s) => s.isLoading);
  const renderMode = useEngramStore((s) => s.renderMode);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const showEdgeLabels = useEngramStore((s) => s.showEdgeLabels);
  const selectNode = useEngramStore((s) => s.selectNode);
  const hoverNode = useEngramStore((s) => s.hoverNode);
  const expandNode = useEngramStore((s) => s.expandNode);
  const fgRef = useRef<{ d3Force: (name: string) => unknown } | null>(null);

  // ForceGraph mutates node/edge objects (adds __threeObj, x, y, z, etc.)
  // so we must give it mutable shallow copies, not frozen Immer proxies.
  const graphData = useMemo(() => {
    const nodeList = Object.values(nodes).map((n) => ({ ...n }));
    const edgeList = Object.values(edges).map((e) => ({ ...e }));
    return { nodes: nodeList, links: edgeList };
  }, [nodes, edges]);

  // Tune force simulation for better layout
  useEffect(() => {
    if (fgRef.current && renderMode === "3d") {
      const fg = fgRef.current as {
        d3Force: (name: string) => { strength?: (v: number) => unknown; distanceMax?: (v: number) => unknown } | undefined;
      };
      const charge = fg.d3Force("charge");
      if (charge?.strength) charge.strength(-40);
      if (charge?.distanceMax) charge.distanceMax(300);
    }
  }, [graphData, renderMode]);

  const handleNodeClick = useCallback(
    (node: { id?: string }) => selectNode(node.id ?? null),
    [selectNode]
  );
  const handleNodeRightClick = useCallback(
    (node: { id?: string }) => {
      if (node.id) expandNode(node.id);
    },
    [expandNode]
  );
  const handleNodeHover = useCallback(
    (node: { id?: string } | null) => hoverNode(node?.id ?? null),
    [hoverNode]
  );

  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => createNodeObject(node, showHeatmap),
    [showHeatmap]
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
          linkWidth={(link: Record<string, unknown>) =>
            Math.max(0.5, ((link.weight as number) ?? 1) * 1.5)
          }
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(34, 211, 238, 0.35)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => "#22d3ee"}
          linkDirectionalParticleSpeed={0.003}
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
          linkColor={() => "rgba(34, 211, 238, 0.35)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => "#22d3ee"}
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
