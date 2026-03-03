import { useMemo } from "react";
import { useEngramStore } from "../store";
import { entityColor } from "../lib/colors";

export function TimelineView() {
  const nodes = useEngramStore((s) => s.nodes);
  const selectNode = useEngramStore((s) => s.selectNode);
  const loadNeighborhood = useEngramStore((s) => s.loadNeighborhood);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);

  const sortedNodes = useMemo(() => {
    return Object.values(nodes)
      .filter((n) => n.createdAt)
      .sort(
        (a, b) =>
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
      );
  }, [nodes]);

  const timeRange = useMemo(() => {
    if (sortedNodes.length === 0)
      return { min: Date.now(), max: Date.now() };
    return {
      min: new Date(sortedNodes[0].createdAt).getTime(),
      max: new Date(sortedNodes[sortedNodes.length - 1].createdAt).getTime(),
    };
  }, [sortedNodes]);

  const handleDotClick = (nodeId: string) => {
    selectNode(nodeId);
    loadNeighborhood(nodeId);
    setCurrentView("graph");
  };

  const monthTicks = useMemo(() => {
    const ticks: Array<{ label: string; position: number }> = [];
    if (sortedNodes.length === 0) return ticks;
    const range = timeRange.max - timeRange.min;
    if (range === 0) {
      ticks.push({
        label: formatMonth(new Date(timeRange.min)),
        position: 50,
      });
      return ticks;
    }
    const seen = new Set<string>();
    for (const node of sortedNodes) {
      const d = new Date(node.createdAt);
      const key = `${d.getFullYear()}-${d.getMonth()}`;
      if (!seen.has(key)) {
        seen.add(key);
        const pos =
          ((d.getTime() - timeRange.min) / range) * 100;
        ticks.push({ label: formatMonth(d), position: pos });
      }
    }
    return ticks;
  }, [sortedNodes, timeRange]);

  if (sortedNodes.length === 0) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span className="label">No entities to display on timeline</span>
      </div>
    );
  }

  const range = timeRange.max - timeRange.min;

  return (
    <div
      className="animate-fade-in"
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-end",
        padding: "0 14px 20px",
        pointerEvents: "none",
      }}
    >
      <div
        className="card"
        style={{
          padding: "16px 20px",
          pointerEvents: "auto",
          position: "relative",
          minHeight: 80,
        }}
      >
        <div className="label" style={{ marginBottom: 16 }}>
          Entity Timeline
        </div>

        {/* Axis */}
        <div style={{ position: "relative", height: 40 }}>
          <div
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: 20,
              height: 1,
              background:
                "linear-gradient(90deg, transparent, var(--border-hover), transparent)",
            }}
          />

          {/* Entity dots */}
          {sortedNodes.map((node) => {
            const color = entityColor(node.entityType);
            const pct =
              range === 0
                ? 50
                : ((new Date(node.createdAt).getTime() - timeRange.min) /
                    range) *
                  100;
            return (
              <button
                key={node.id}
                onClick={() => handleDotClick(node.id)}
                title={`${node.name} (${node.entityType})`}
                style={{
                  position: "absolute",
                  left: `${pct}%`,
                  top: 14,
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: color,
                  border: "2px solid var(--surface-solid)",
                  transform: "translateX(-50%)",
                  cursor: "pointer",
                  transition: "all 0.15s",
                  boxShadow: `0 0 6px ${color}40`,
                  padding: 0,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform =
                    "translateX(-50%) scale(1.5)";
                  e.currentTarget.style.boxShadow = `0 0 12px ${color}80`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform =
                    "translateX(-50%) scale(1)";
                  e.currentTarget.style.boxShadow = `0 0 6px ${color}40`;
                }}
              />
            );
          })}

          {/* Month ticks */}
          {monthTicks.map((tick) => (
            <div
              key={tick.label}
              style={{
                position: "absolute",
                left: `${tick.position}%`,
                top: 28,
                transform: "translateX(-50%)",
              }}
            >
              <span
                className="mono"
                style={{
                  fontSize: 9,
                  color: "var(--text-muted)",
                  whiteSpace: "nowrap",
                }}
              >
                {tick.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function formatMonth(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}
