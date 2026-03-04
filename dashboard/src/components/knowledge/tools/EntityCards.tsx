import { useEngramStore } from "../../../store";
import { entityColor, entityColorDim } from "../../../lib/colors";
import { activationColor, activationGlow } from "../../../lib/colors";

interface Entity {
  id: string;
  name: string;
  entityType: string;
  summary?: string | null;
  score: number;
  activation: number;
}

export function EntityCards({ entities }: { entities: Entity[] }) {
  const openDrawer = useEngramStore((s) => s.openDrawer);

  if (!entities || entities.length === 0) return null;

  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        overflowX: "auto",
        padding: "4px 0",
        scrollbarWidth: "thin",
      }}
    >
      {entities.map((ent) => {
        const color = entityColor(ent.entityType);
        return (
          <button
            key={ent.id}
            onClick={() => openDrawer(ent.id)}
            style={{
              flexShrink: 0,
              width: 160,
              padding: "10px 12px",
              borderRadius: "var(--radius-sm)",
              border: `1px solid ${color}20`,
              background: entityColorDim(ent.entityType, 0.06),
              cursor: "pointer",
              textAlign: "left",
              fontFamily: "var(--font-body)",
              display: "flex",
              flexDirection: "column",
              gap: 6,
              transition: "all 0.15s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = `${color}40`;
              e.currentTarget.style.background = entityColorDim(ent.entityType, 0.12);
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = `${color}20`;
              e.currentTarget.style.background = entityColorDim(ent.entityType, 0.06);
            }}
          >
            {/* Header: type badge */}
            <span
              style={{
                fontSize: 9,
                fontFamily: "var(--font-mono)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color,
                opacity: 0.8,
              }}
            >
              {ent.entityType}
            </span>

            {/* Name */}
            <span
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: "var(--text-primary)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {ent.name}
            </span>

            {/* Summary */}
            {ent.summary && (
              <span
                style={{
                  fontSize: 10,
                  color: "var(--text-secondary)",
                  lineHeight: 1.3,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical",
                }}
              >
                {ent.summary}
              </span>
            )}

            {/* Activation glow bar */}
            <div
              style={{
                height: 3,
                borderRadius: 2,
                background: "rgba(255,255,255,0.05)",
                marginTop: "auto",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${Math.min(100, ent.activation * 100)}%`,
                  background: activationColor(ent.activation),
                  boxShadow: `0 0 6px ${activationGlow(ent.activation, 0.5)}`,
                  borderRadius: 2,
                  transition: "width 0.3s ease",
                }}
              />
            </div>

            {/* Score */}
            <span
              className="mono"
              style={{ fontSize: 9, color: "var(--text-muted)" }}
            >
              {(ent.score * 100).toFixed(0)}% match
            </span>
          </button>
        );
      })}
    </div>
  );
}
