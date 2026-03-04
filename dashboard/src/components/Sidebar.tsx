import { useEngramStore } from "../store";
import type { DashboardView } from "../store/types";
import { SearchBar } from "./SearchBar";
import { ConnectionStatus } from "./ConnectionStatus";

const VIEWS: { id: DashboardView; label: string; icon: string }[] = [
  { id: "knowledge", label: "Knowledge", icon: "\u25C8" },
  { id: "graph", label: "Graph", icon: "\u25C9" },
  { id: "timeline", label: "Timeline", icon: "\u25F7" },
  { id: "feed", label: "Feed", icon: "\u2630" },
  { id: "activation", label: "Activation", icon: "\u26A1" },
  { id: "stats", label: "Stats", icon: "\u25A4" },
  { id: "consolidation", label: "Consolidate", icon: "\u27F3" },
];

export function Sidebar() {
  const currentView = useEngramStore((s) => s.currentView);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);

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
          Memory Explorer
        </p>
      </div>

      <div className="accent-bar" style={{ margin: "14px 18px 10px" }} />

      {/* Search */}
      <div style={{ padding: "0 10px" }}>
        <SearchBar />
      </div>

      {/* Navigation */}
      <nav style={{ padding: "10px 10px 0", flex: 1 }}>
        {VIEWS.map((v) => {
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
                  ? "rgba(34, 211, 238, 0.08)"
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
