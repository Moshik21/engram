import { useEngramStore } from "../store";
import type { DashboardView, DashboardMode } from "../store/types";
import { SearchBar } from "./SearchBar";
import { ConnectionStatus } from "./ConnectionStatus";

const OBSERVATORY_VIEWS: { id: DashboardView; label: string; icon: string }[] = [
  { id: "lifecycle", label: "Brain Loop", icon: "\u27F2" },
  { id: "knowledge", label: "Knowledge", icon: "\u25C8" },
  { id: "graph", label: "Graph", icon: "\u25C9" },
  { id: "timeline", label: "Timeline", icon: "\u25F7" },
  { id: "feed", label: "Feed", icon: "\u2630" },
  { id: "activation", label: "Activation", icon: "\u26A1" },
  { id: "stats", label: "Stats", icon: "\u25A4" },
  { id: "evaluation", label: "Evaluate", icon: "\u25CE" },
  { id: "consolidation", label: "Consolidate", icon: "\u27F3" },
];

const NERVE_VIEWS: { id: DashboardView; label: string; icon: string }[] = [
  { id: "nerve_center", label: "Nerve Center", icon: "\u2699" },
  { id: "neural_field", label: "Neural Field", icon: "\u25C9" },
  { id: "ingestion", label: "Ingestion Chamber", icon: "\u269B" },
  { id: "adjudicate", label: "Adjudication", icon: "\u2696" },
  { id: "immunity", label: "Immunity Sweep", icon: "\u26E8" },
  { id: "profile", label: "Cerebral Profile", icon: "\u2606" },
  { id: "synaptic_log", label: "Plasticity Log", icon: "\u2721" },
];

// Map between modes for toggle
const VIEW_MAP: Record<string, DashboardView> = {
  lifecycle: "nerve_center",
  nerve_center: "lifecycle",
  knowledge: "nerve_center",
  graph: "neural_field",
  neural_field: "graph",
  feed: "ingestion",
  ingestion: "feed",
  stats: "profile",
  evaluation: "profile",
  profile: "stats",
  consolidation: "synaptic_log",
  synaptic_log: "consolidation",
};

export function Sidebar() {
  const currentView = useEngramStore((s) => s.currentView);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);
  const dashboardMode = useEngramStore((s) => s.dashboardMode);
  const setDashboardMode = useEngramStore((s) => s.setDashboardMode);

  const views = dashboardMode === "nerve" ? NERVE_VIEWS : OBSERVATORY_VIEWS;

  const handleModeToggle = () => {
    const newMode: DashboardMode = dashboardMode === "observatory" ? "nerve" : "observatory";
    setDashboardMode(newMode);
    // Switch to equivalent view in new mode
    const mappedView = VIEW_MAP[currentView];
    if (mappedView) {
      setCurrentView(mappedView);
    } else {
      // Default to first view in new mode
      const defaultView = newMode === "nerve" ? "nerve_center" : "lifecycle";
      setCurrentView(defaultView);
    }
  };

  return (
    <aside
      className="glass-elevated animate-slide-left"
      style={{
        width: 220,
        borderRadius: "var(--radius-lg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Brand */}
      <div style={{ padding: "20px 18px 0" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          <h1
            className="display"
            style={{
              fontSize: 26,
              fontWeight: 400,
              color: "#fff",
              letterSpacing: "-0.02em",
              lineHeight: 1,
            }}
          >
            Engram
          </h1>
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--accent)",
              boxShadow: "0 0 8px var(--accent-glow-strong)",
              display: "inline-block",
              flexShrink: 0,
            }}
          />
        </div>
        <p className="label" style={{ marginTop: 3, fontSize: 9, letterSpacing: "0.1em" }}>
          {dashboardMode === "nerve" ? "Nerve Center" : "Memory Explorer"}
        </p>
      </div>

      {/* Mode Toggle */}
      <div style={{ padding: "10px 18px 0" }}>
        <button
          onClick={handleModeToggle}
          style={{
            width: "100%",
            padding: "6px 10px",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            background: dashboardMode === "nerve" ? "rgba(103, 232, 249, 0.08)" : "rgba(103, 232, 249, 0.04)",
            color: dashboardMode === "nerve" ? "var(--accent)" : "var(--text-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            fontWeight: 500,
            letterSpacing: "0.06em",
            textTransform: "uppercase" as const,
            cursor: "pointer",
            transition: "all 0.2s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
          }}
        >
          <span>{dashboardMode === "nerve" ? "\u2630" : "\u2699"}</span>
          {dashboardMode === "nerve" ? "Observatory" : "Nerve Center"}
        </button>
      </div>

      <div className="accent-bar" style={{ margin: "14px 18px 10px" }} />

      {/* Search */}
      <div style={{ padding: "0 10px" }}>
        <SearchBar />
      </div>

      {/* Navigation */}
      <nav style={{ padding: "10px 10px 0", flex: 1 }}>
        {views.map((v) => {
          const active = currentView === v.id;
          return (
            <button
              key={v.id}
              onClick={() => setCurrentView(v.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 9,
                width: "100%",
                padding: "7px 10px",
                marginBottom: 1,
                border: "none",
                borderRadius: "var(--radius-sm)",
                background: active
                  ? dashboardMode === "nerve"
                    ? "rgba(212, 168, 75, 0.08)"
                    : "rgba(34, 211, 238, 0.08)"
                  : "transparent",
                color: active ? "var(--accent)" : "var(--text-secondary)",
                fontFamily: "var(--font-body)",
                fontSize: 13,
                fontWeight: active ? 500 : 400,
                cursor: "pointer",
                transition: "all 0.15s ease",
                textAlign: "left",
                position: "relative",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                  e.currentTarget.style.color = "var(--text-primary)";
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }
              }}
            >
              {/* Active indicator line */}
              {active && (
                <span
                  style={{
                    position: "absolute",
                    left: 0,
                    top: "50%",
                    transform: "translateY(-50%)",
                    width: 2,
                    height: 16,
                    borderRadius: 1,
                    background: "var(--accent)",
                    boxShadow: "0 0 6px var(--accent-glow-strong)",
                  }}
                />
              )}
              <span
                style={{
                  fontSize: 13,
                  opacity: active ? 1 : 0.4,
                  width: 16,
                  textAlign: "center",
                  flexShrink: 0,
                }}
              >
                {v.icon}
              </span>
              <span>{v.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Connection status */}
      <div style={{ borderTop: "1px solid var(--border)", padding: "10px 14px" }}>
        <ConnectionStatus />
      </div>
    </aside>
  );
}
