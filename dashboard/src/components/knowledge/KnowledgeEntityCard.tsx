import { useEngramStore } from "../../store";
import { entityColor, entityColorDim } from "../../lib/colors";
import type { SearchResult } from "../../store/types";

export function KnowledgeEntityCard({ entity }: { entity: SearchResult }) {
  const expandedEntityId = useEngramStore((s) => s.expandedEntityId);
  const entityDetail = useEngramStore((s) => s.entityDetail);
  const expandEntity = useEngramStore((s) => s.expandEntity);
  const selectNode = useEngramStore((s) => s.selectNode);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);
  const openDrawer = useEngramStore((s) => s.openDrawer);
  const setBrowseOverlayOpen = useEngramStore((s) => s.setBrowseOverlayOpen);

  const isExpanded = expandedEntityId === entity.id;
  const color = entityColor(entity.entityType);
  const activation = entity.activationScore;
  const pct = Math.min(activation * 100, 100);

  return (
    <div
      className="card card-glow"
      onClick={() => expandEntity(isExpanded ? null : entity.id)}
      style={{
        padding: "10px 12px",
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: color,
            boxShadow: `0 0 6px ${color}40`,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {entity.name}
        </span>
      </div>

      {/* Summary */}
      {entity.summary && (
        <p style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 4, lineHeight: 1.4, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: isExpanded ? undefined : 2, WebkitBoxOrient: "vertical" }}>
          {entity.summary}
        </p>
      )}

      {/* Activation bar */}
      <div className="metric-bar" style={{ marginTop: 6 }}>
        <div
          className="metric-bar-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}99, ${color})`,
          }}
        />
      </div>

      {/* Expanded detail */}
      {isExpanded && entityDetail && entityDetail.id === entity.id && (
        <div
          className="animate-slide-up"
          style={{ marginTop: 10, borderTop: "1px solid var(--border)", paddingTop: 8 }}
        >
          {entityDetail.facts.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <span className="label" style={{ marginBottom: 2 }}>Relationships</span>
              {entityDetail.facts.slice(0, 8).map((fact) => (
                <div
                  key={fact.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                  }}
                >
                  <span style={{ color: entityColor(fact.other.entityType), fontSize: 9 }}>
                    {fact.direction === "outgoing" ? "\u2192" : "\u2190"}
                  </span>
                  <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)" }}>
                    {fact.predicate}
                  </span>
                  <span style={{ color: entityColor(fact.other.entityType) }}>
                    {fact.other.name}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                selectNode(entity.id);
                setCurrentView("graph");
              }}
              style={{
                flex: 1,
                padding: "5px 0",
                background: entityColorDim(entity.entityType, 0.08),
                border: `1px solid ${color}20`,
                borderRadius: "var(--radius-xs)",
                color,
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                fontWeight: 500,
                cursor: "pointer",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              Graph
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                openDrawer(entity.id);
                setBrowseOverlayOpen(false);
              }}
              style={{
                flex: 1,
                padding: "5px 0",
                background: "rgba(34, 211, 238, 0.06)",
                border: "1px solid var(--border-active)",
                borderRadius: "var(--radius-xs)",
                color: "var(--accent)",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                fontWeight: 500,
                cursor: "pointer",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              Details
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
