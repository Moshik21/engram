import { useRef, useCallback } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { entityColor } from "../../../lib/colors";

interface GraphNode {
  id: string;
  name: string;
  type: string;
}

interface GraphEdge {
  source: string;
  target: string;
  predicate: string;
  weight: number;
}

interface GraphData {
  centralEntity: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function RelationshipMiniGraph({ data }: { data: GraphData }) {
  const containerRef = useRef<HTMLDivElement>(null);

  const graphData = {
    nodes: data.nodes.map((n) => ({ ...n, color: entityColor(n.type) })),
    links: data.edges.map((e) => ({
      source: e.source,
      target: e.target,
      label: e.predicate,
      weight: e.weight,
    })),
  };

  const nodeCanvasObject = useCallback(
    (node: Record<string, unknown>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const label = (node.name as string) || "";
      const fontSize = Math.max(10 / globalScale, 2);
      const r = node.id === data.nodes.find((n) => n.name === data.centralEntity)?.id ? 6 : 4;
      const color = (node.color as string) || "#94a3b8";

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x as number, node.y as number, r, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = color + "60";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(label, node.x as number, (node.y as number) + r + 2);
    },
    [data.centralEntity, data.nodes],
  );

  const linkCanvasObject = useCallback(
    (link: Record<string, unknown>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const source = link.source as Record<string, number>;
      const target = link.target as Record<string, number>;
      if (!source?.x || !target?.x) return;

      // Line
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Label at midpoint
      const fontSize = Math.max(8 / globalScale, 1.5);
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
      ctx.fillText((link.label as string) || "", midX, midY);
    },
    [],
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: 200,
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border)",
        background: "rgba(0, 0, 0, 0.2)",
        overflow: "hidden",
      }}
    >
      <ForceGraph2D
        graphData={graphData}
        width={400}
        height={200}
        backgroundColor="transparent"
        nodeCanvasObject={nodeCanvasObject}
        linkCanvasObject={linkCanvasObject}
        cooldownTicks={60}
        enableZoomInteraction={false}
        enablePanInteraction={false}
      />
    </div>
  );
}
