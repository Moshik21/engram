import { useEngramStore } from "../store";
import { useNodeCount, useEdgeCount, useNodeById } from "../store/graphSelectors";
import { GraphControls } from "./GraphControls";
import { TimeScrubber } from "./TimeScrubber";

export function TopBar() {
  const centerNodeId = useEngramStore((s) => s.centerNodeId);
  const brainMapScope = useEngramStore((s) => s.brainMapScope);
  const representation = useEngramStore((s) => s.representation);
  const activeRegionId = useEngramStore((s) => s.activeRegionId);
  const atlasHistory = useEngramStore((s) => s.atlasHistory);
  const atlasSnapshotId = useEngramStore((s) => s.atlasSnapshotId);
  const regionData = useEngramStore((s) => s.regionData);
  const nodeCount = useNodeCount();
  const edgeCount = useEdgeCount();
  const currentView = useEngramStore((s) => s.currentView);
  const loadAtlas = useEngramStore((s) => s.loadAtlas);
  const loadRegion = useEngramStore((s) => s.loadRegion);
  const centerNode = useNodeById(centerNodeId);

  const showGraphControls =
    (currentView === "graph" || currentView === "timeline") &&
    (brainMapScope === "neighborhood" || brainMapScope === "temporal");
  const showTimeScrubber =
    currentView === "graph" &&
    (showGraphControls ||
      ((brainMapScope === "atlas" || brainMapScope === "region") &&
        atlasHistory.length > 1));

  const scopeLabel =
    brainMapScope === "atlas"
      ? "Atlas"
      : brainMapScope === "region"
        ? "Region"
      : brainMapScope === "temporal"
        ? "Temporal"
        : "Neighborhood";

  const usesAbstractCounts =
    representation?.scope === "atlas" || representation?.scope === "region";

  const primaryCount =
    usesAbstractCounts
      ? representation.displayedNodeCount
      : nodeCount;

  const primaryLabel =
    representation?.scope === "atlas"
      ? "regions"
      : representation?.scope === "region"
        ? "facets"
        : "nodes";

  const secondaryCount =
    usesAbstractCounts
      ? representation.representedEntityCount
      : edgeCount;

  const secondaryLabel =
    usesAbstractCounts ? "memories" : "edges";

  return (
    <div className="flex items-center gap-2" style={{ height: 40 }}>
      {(currentView === "graph" || currentView === "timeline") && representation && (
        <div
          className="card animate-fade-in"
          style={{
            borderRadius: "var(--radius-md)",
            padding: "5px 12px",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span
            className="mono"
            style={{
              fontSize: 10,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--accent)",
            }}
          >
            {scopeLabel}
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {representation.layout}
          </span>
        </div>
      )}

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

      {currentView === "graph" && brainMapScope !== "atlas" && activeRegionId && regionData && brainMapScope !== "region" && (
        <button
          type="button"
          className="card animate-fade-in"
          onClick={() => {
            void loadRegion(activeRegionId, {
              snapshotId: atlasSnapshotId,
            });
          }}
          style={{
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border)",
            background: "var(--surface)",
            padding: "5px 12px",
            display: "flex",
            alignItems: "center",
            gap: 7,
            color: "var(--text-primary)",
            cursor: "pointer",
          }}
        >
          <span
            className="mono"
            style={{ fontSize: 10, color: "var(--accent)", letterSpacing: "0.12em" }}
          >
            Region
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {regionData.region.label}
          </span>
        </button>
      )}

      {currentView === "graph" && brainMapScope !== "atlas" && (
        <button
          type="button"
          className="card animate-fade-in"
          onClick={() => {
            void loadAtlas({
              snapshotId: atlasSnapshotId,
            });
          }}
          style={{
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border)",
            background: "var(--surface)",
            padding: "5px 12px",
            display: "flex",
            alignItems: "center",
            gap: 7,
            color: "var(--text-primary)",
            cursor: "pointer",
          }}
        >
          <span
            className="mono"
            style={{ fontSize: 10, color: "var(--accent)", letterSpacing: "0.12em" }}
          >
            Atlas
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            back to overview
          </span>
        </button>
      )}

      {/* Center node indicator */}
      {centerNode &&
        currentView === "graph" &&
        (brainMapScope === "neighborhood" || brainMapScope === "temporal") && (
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
      {!showGraphControls && showTimeScrubber && <TimeScrubber />}

      {currentView === "graph" && atlasSnapshotId && (
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
          <span
            className="mono"
            style={{ fontSize: 10, color: "var(--warning)", letterSpacing: "0.12em" }}
          >
            Snapshot
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            historical atlas view
          </span>
        </div>
      )}

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
          {primaryCount}
        </span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {primaryLabel}
        </span>
        <span style={{ fontSize: 11, color: "var(--text-ghost)", margin: "0 2px" }}>
          ·
        </span>
        <span
          className="mono tabular-nums"
          style={{ fontSize: 11, color: "var(--info)", fontWeight: 500 }}
        >
          {secondaryCount}
        </span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {secondaryLabel}
        </span>
      </div>
    </div>
  );
}
