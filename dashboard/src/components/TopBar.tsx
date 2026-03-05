import { useEngramStore } from "../store";
import { useNodeCount, useEdgeCount, useNodeById } from "../store/graphSelectors";
import { GraphControls } from "./GraphControls";
import { TimeScrubber } from "./TimeScrubber";

export function TopBar() {
  const centerNodeId = useEngramStore((s) => s.centerNodeId);
  const nodeCount = useNodeCount();
  const edgeCount = useEdgeCount();
  const currentView = useEngramStore((s) => s.currentView);
  const centerNode = useNodeById(centerNodeId);

  const showGraphControls =
    currentView === "graph" || currentView === "timeline";

  return (
    <div className="flex items-center gap-2" style={{ height: 40 }}>
      {/* Graph controls pill */}
      {showGraphControls && (
        <div
          className="card"
          style={{
            borderRadius: "var(--radius-md)",
            padding: "5px 10px",
          }}
        >
          <GraphControls />
        </div>
      )}

      {/* Center node indicator */}
      {centerNode && currentView === "graph" && (
        <div
          className="card animate-fade-in"
          style={{
            borderRadius: "var(--radius-md)",
            padding: "5px 12px",
            display: "flex",
            alignItems: "center",
            gap: 7,
          }}
        >
          <div
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--accent)",
              boxShadow: "0 0 6px var(--accent-glow-strong)",
            }}
          />
          <span
            style={{
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            centered on
          </span>
          <span
            className="display"
            style={{ fontSize: 14, color: "#fff" }}
          >
            {centerNode.name}
          </span>
        </div>
      )}

      <div style={{ flex: 1 }} />

      {/* Time scrubber */}
      {showGraphControls && <TimeScrubber />}

      {/* Node count */}
      <div
        className="card"
        style={{
          borderRadius: "var(--radius-md)",
          padding: "5px 12px",
          display: "flex",
          alignItems: "center",
          gap: 4,
        }}
      >
        <span
          className="mono tabular-nums"
          style={{ fontSize: 11, color: "var(--accent)", fontWeight: 500 }}
        >
          {nodeCount}
        </span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          nodes
        </span>
        <span style={{ fontSize: 11, color: "var(--text-ghost)", margin: "0 2px" }}>
          ·
        </span>
        <span
          className="mono tabular-nums"
          style={{ fontSize: 11, color: "var(--info)", fontWeight: 500 }}
        >
          {edgeCount}
        </span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          edges
        </span>
      </div>
    </div>
  );
}
