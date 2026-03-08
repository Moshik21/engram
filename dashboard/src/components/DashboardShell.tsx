import { Suspense, lazy } from "react";
import { useEngramStore } from "../store";
import { useWebSocket } from "../hooks/useWebSocket";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const BrainMapPanel = lazy(() =>
  import("./BrainMapPanel").then((module) => ({
    default: module.BrainMapPanel,
  })),
);
const NodeDetailPanel = lazy(() =>
  import("./NodeDetailPanel").then((module) => ({
    default: module.NodeDetailPanel,
  })),
);
const MemoryFeed = lazy(() =>
  import("./MemoryFeed").then((module) => ({
    default: module.MemoryFeed,
  })),
);
const StatsPanel = lazy(() =>
  import("./StatsPanel").then((module) => ({
    default: module.StatsPanel,
  })),
);
const TimelineView = lazy(() =>
  import("./TimelineView").then((module) => ({
    default: module.TimelineView,
  })),
);
const ActivationMonitor = lazy(() =>
  import("./ActivationMonitor").then((module) => ({
    default: module.ActivationMonitor,
  })),
);
const ConsolidationPanel = lazy(() =>
  import("./ConsolidationPanel").then((module) => ({
    default: module.ConsolidationPanel,
  })),
);
const KnowledgePanel = lazy(() =>
  import("./knowledge/KnowledgePanel").then((module) => ({
    default: module.KnowledgePanel,
  })),
);

function PanelFallback() {
  return (
    <div
      className="absolute inset-0 flex items-center justify-center"
      style={{ top: 54 }}
    >
      <div
        className="card"
        style={{
          padding: "12px 18px",
          borderColor: "var(--border-hover)",
        }}
      >
        <span className="label">Loading view...</span>
      </div>
    </div>
  );
}

export function DashboardShell() {
  const error = useEngramStore((s) => s.error);
  const currentView = useEngramStore((s) => s.currentView);
  const brainMapScope = useEngramStore((s) => s.brainMapScope);
  const showNodeDetailPanel =
    currentView === "graph" &&
    (brainMapScope === "neighborhood" || brainMapScope === "temporal");

  useWebSocket();

  const sidebarWidth = 232;
  const gap = 10;
  const contentLeft = sidebarWidth + gap * 2 + gap;

  function renderMainContent() {
    switch (currentView) {
      case "feed":
        return (
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ left: contentLeft, top: 54 }}
          >
            <MemoryFeed />
          </div>
        );
      case "stats":
        return (
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ left: contentLeft, top: 54 }}
          >
            <StatsPanel />
          </div>
        );
      case "timeline":
        return (
          <>
            <div className="absolute inset-0">
              <BrainMapPanel />
            </div>
            <div className="vignette" />
            <div
              className="absolute inset-0 z-[5]"
              style={{ left: contentLeft, top: 54 }}
            >
              <TimelineView />
            </div>
          </>
        );
      case "activation":
        return (
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ left: contentLeft, top: 54 }}
          >
            <ActivationMonitor />
          </div>
        );
      case "consolidation":
        return (
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ left: contentLeft, top: 54 }}
          >
            <ConsolidationPanel />
          </div>
        );
      case "knowledge":
        return (
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ left: contentLeft, top: 54 }}
          >
            <KnowledgePanel />
          </div>
        );
      case "graph":
      default:
        return (
          <>
            <div className="absolute inset-0">
              <BrainMapPanel />
            </div>
            <div className="vignette" />
            {showNodeDetailPanel && <NodeDetailPanel />}
          </>
        );
    }
  }

  return (
    <div
      className="relative h-screen w-screen overflow-hidden"
      style={{ background: "var(--void)" }}
    >
      <Suspense fallback={<PanelFallback />}>{renderMainContent()}</Suspense>

      {/* Floating sidebar */}
      <div
        className="absolute z-10"
        style={{ left: gap, top: gap, bottom: gap }}
      >
        <Sidebar />
      </div>

      {/* Floating top controls */}
      <div
        className="absolute z-10 animate-fade-in"
        style={{ top: gap, left: contentLeft, right: gap }}
      >
        <TopBar />
      </div>

      {/* Error toast */}
      {error && (
        <div
          className="card absolute left-1/2 top-16 z-30 -translate-x-1/2 animate-slide-down"
          style={{
            borderColor: "rgba(248, 113, 113, 0.15)",
            padding: "8px 16px",
          }}
        >
          <span className="mono" style={{ color: "var(--danger)", fontSize: 12 }}>
            {error}
          </span>
        </div>
      )}
    </div>
  );
}
