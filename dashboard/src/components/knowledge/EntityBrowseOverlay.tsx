import { useEngramStore } from "../../store";
import { EntityGroupSection } from "./EntityGroupSection";

const TYPE_ORDER = ["Person", "Technology", "Project", "Organization", "Concept", "Location", "Event"];

export function EntityBrowseOverlay() {
  const entityGroups = useEngramStore((s) => s.entityGroups);
  const setBrowseOverlayOpen = useEngramStore((s) => s.setBrowseOverlayOpen);

  const close = () => setBrowseOverlayOpen(false);

  const sortedTypes = Object.keys(entityGroups).sort((a, b) => {
    const ai = TYPE_ORDER.indexOf(a);
    const bi = TYPE_ORDER.indexOf(b);
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
  });

  return (
    <div
      onClick={close}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 50,
        background: "rgba(3, 4, 8, 0.7)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        className="glass-elevated animate-fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "100%",
          maxWidth: 800,
          maxHeight: "80vh",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "14px 18px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)" }}>
            Browse All Entities
          </span>
          <button
            onClick={close}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: 16,
              lineHeight: 1,
              padding: "2px 4px",
            }}
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div style={{ overflowY: "auto", flex: 1, padding: "16px 18px" }}>
          {sortedTypes.length === 0 ? (
            <div style={{ textAlign: "center", padding: 40, color: "var(--text-muted)", fontSize: 12 }}>
              No entities stored yet
            </div>
          ) : (
            sortedTypes.map((type) => (
              <EntityGroupSection key={type} type={type} entities={entityGroups[type]} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
