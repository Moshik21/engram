import { useState, useRef, useCallback } from "react";
import { useEngramStore } from "../store";
import { generateSyntheticGraph } from "../lib/syntheticGraph";
import { FrameMeter, type BenchmarkResult } from "../lib/frameMeter";
import type { GraphNode, GraphEdge } from "../store/types";

const DEFAULT_TIERS = [100, 500, 1000, 2000, 5000];

function fpsColor(fps: number): string {
  if (fps >= 55) return "#22c55e";
  if (fps >= 30) return "#eab308";
  return "#ef4444";
}

export default function StressTestPanel() {
  const [tiers, setTiers] = useState(DEFAULT_TIERS.join(", "));
  const [duration, setDuration] = useState(5000);
  const [warmup, setWarmup] = useState(2000);
  const [edgesPerNode, setEdgesPerNode] = useState(2.5);
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [running, setRunning] = useState(false);
  const [currentTier, setCurrentTier] = useState<number | null>(null);

  const meterRef = useRef<FrameMeter | null>(null);
  const cancelledRef = useRef(false);
  const savedGraphRef = useRef<{
    nodes: Record<string, GraphNode>;
    edges: Record<string, GraphEdge>;
  } | null>(null);

  const restore = useCallback(() => {
    if (savedGraphRef.current) {
      useEngramStore.setState({
        nodes: savedGraphRef.current.nodes,
        edges: savedGraphRef.current.edges,
      });
      savedGraphRef.current = null;
    }
  }, []);

  const runRamp = useCallback(async () => {
    const tierList = tiers
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n) && n > 0);
    if (tierList.length === 0) return;

    setRunning(true);
    setResults([]);
    cancelledRef.current = false;

    const state = useEngramStore.getState();
    savedGraphRef.current = { nodes: { ...state.nodes }, edges: { ...state.edges } };

    const collected: BenchmarkResult[] = [];

    for (const count of tierList) {
      if (cancelledRef.current) break;

      setCurrentTier(count);

      const { nodes, edges } = generateSyntheticGraph(count, edgesPerNode);
      useEngramStore.setState({ nodes, edges });

      // Let React render + force layout settle for a frame
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

      const meter = new FrameMeter(count, duration, warmup);
      meterRef.current = meter;

      try {
        const result = await meter.start();
        collected.push(result);
        setResults([...collected]);
      } catch {
        // cancelled
        break;
      }
    }

    restore();
    setRunning(false);
    setCurrentTier(null);
  }, [tiers, duration, warmup, edgesPerNode, restore]);

  const handleCancel = useCallback(() => {
    cancelledRef.current = true;
    meterRef.current?.cancel();
    restore();
    setRunning(false);
    setCurrentTier(null);
  }, [restore]);

  const maxSustainable = results.length > 0
    ? results.filter((r) => r.avgFps >= 30).sort((a, b) => b.nodeCount - a.nodeCount)[0]
    : null;

  return (
    <div
      style={{
        position: "absolute",
        top: 56,
        right: 12,
        width: 420,
        background: "rgba(10, 12, 20, 0.95)",
        border: "1px solid rgba(100, 160, 255, 0.2)",
        borderRadius: 8,
        padding: 16,
        zIndex: 100,
        color: "#e2e8f0",
        fontFamily: "var(--font-mono, monospace)",
        fontSize: 12,
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 12, fontSize: 14 }}>
        Stress Test
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
        <label>
          <span style={{ opacity: 0.7 }}>Node tiers</span>
          <input
            value={tiers}
            onChange={(e) => setTiers(e.target.value)}
            disabled={running}
            style={inputStyle}
          />
        </label>
        <label>
          <span style={{ opacity: 0.7 }}>Edges/node</span>
          <input
            type="number"
            step={0.5}
            value={edgesPerNode}
            onChange={(e) => setEdgesPerNode(parseFloat(e.target.value) || 2.5)}
            disabled={running}
            style={inputStyle}
          />
        </label>
        <label>
          <span style={{ opacity: 0.7 }}>Duration (ms)</span>
          <input
            type="number"
            value={duration}
            onChange={(e) => setDuration(parseInt(e.target.value, 10) || 5000)}
            disabled={running}
            style={inputStyle}
          />
        </label>
        <label>
          <span style={{ opacity: 0.7 }}>Warmup (ms)</span>
          <input
            type="number"
            value={warmup}
            onChange={(e) => setWarmup(parseInt(e.target.value, 10) || 2000)}
            disabled={running}
            style={inputStyle}
          />
        </label>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {!running ? (
          <button onClick={runRamp} style={btnStyle}>
            Run Ramp
          </button>
        ) : (
          <button onClick={handleCancel} style={{ ...btnStyle, background: "#7f1d1d" }}>
            Cancel
          </button>
        )}
        {running && currentTier !== null && (
          <span style={{ opacity: 0.7, alignSelf: "center" }}>
            Testing {currentTier} nodes...
          </span>
        )}
      </div>

      {results.length > 0 && (
        <>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid rgba(100, 160, 255, 0.2)" }}>
                <th style={thStyle}>Nodes</th>
                <th style={thStyle}>Avg FPS</th>
                <th style={thStyle}>Min FPS</th>
                <th style={thStyle}>P95 ms</th>
                <th style={thStyle}>Dropped</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.nodeCount}>
                  <td style={tdStyle}>{r.nodeCount}</td>
                  <td style={{ ...tdStyle, color: fpsColor(r.avgFps) }}>
                    {r.avgFps.toFixed(1)}
                  </td>
                  <td style={{ ...tdStyle, color: fpsColor(r.minFps) }}>
                    {r.minFps.toFixed(1)}
                  </td>
                  <td style={tdStyle}>{r.p95FrameTime.toFixed(1)}</td>
                  <td style={tdStyle}>{r.droppedFrames}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {maxSustainable && (
            <div style={{ marginTop: 8, opacity: 0.8 }}>
              Max sustainable:{" "}
              <span style={{ color: "#22c55e", fontWeight: 700 }}>
                {maxSustainable.nodeCount} nodes
              </span>{" "}
              @ {maxSustainable.avgFps.toFixed(0)} fps
            </div>
          )}
        </>
      )}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  display: "block",
  width: "100%",
  marginTop: 2,
  padding: "4px 6px",
  background: "rgba(30, 35, 50, 0.8)",
  border: "1px solid rgba(100, 160, 255, 0.15)",
  borderRadius: 4,
  color: "#e2e8f0",
  fontSize: 12,
  fontFamily: "inherit",
};

const btnStyle: React.CSSProperties = {
  padding: "6px 16px",
  background: "rgba(100, 160, 255, 0.15)",
  border: "1px solid rgba(100, 160, 255, 0.3)",
  borderRadius: 4,
  color: "#e2e8f0",
  cursor: "pointer",
  fontSize: 12,
  fontFamily: "inherit",
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "4px 6px",
  opacity: 0.7,
};

const tdStyle: React.CSSProperties = {
  padding: "4px 6px",
};
