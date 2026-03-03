import { useEngramStore } from "../store";
import { entityColor, activationColor } from "../lib/colors";
import { formatRelativeTime } from "../lib/utils";

export function NodeTooltip() {
  const hoveredNodeId = useEngramStore((s) => s.hoveredNodeId);
  const nodes = useEngramStore((s) => s.nodes);

  if (!hoveredNodeId) return null;
  const node = nodes[hoveredNodeId];
  if (!node) return null;

  const typeColor = entityColor(node.entityType);

  return (
    <div
      className="card animate-fade-in"
      style={{
        position: "absolute",
        right: 14,
        top: 56,
        zIndex: 20,
        width: 200,
        borderRadius: "var(--radius-md)",
        padding: 12,
        pointerEvents: "none",
        borderColor: `${typeColor}20`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          marginBottom: 6,
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: typeColor,
            boxShadow: `0 0 6px ${typeColor}50`,
            flexShrink: 0,
          }}
        />
        <span
          className="display"
          style={{ fontSize: 15, color: "#fff" }}
        >
          {node.name}
        </span>
      </div>

      <span
        className="mono"
        style={{
          fontSize: 9,
          color: typeColor,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
        }}
      >
        {node.entityType}
      </span>

      <div
        style={{
          marginTop: 8,
          display: "flex",
          flexDirection: "column",
          gap: 3,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 11,
          }}
        >
          <span style={{ color: "var(--text-secondary)" }}>Activation</span>
          <span
            className="mono tabular-nums"
            style={{
              color: activationColor(node.activationCurrent),
              fontWeight: 500,
            }}
          >
            {(node.activationCurrent * 100).toFixed(0)}%
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 11,
          }}
        >
          <span style={{ color: "var(--text-secondary)" }}>Accesses</span>
          <span className="mono tabular-nums" style={{ color: "var(--text-primary)" }}>
            {node.accessCount}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 11,
          }}
        >
          <span style={{ color: "var(--text-secondary)" }}>Last seen</span>
          <span className="mono" style={{ color: "var(--text-primary)" }}>
            {formatRelativeTime(node.lastAccessed)}
          </span>
        </div>
      </div>
    </div>
  );
}
