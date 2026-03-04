import { useState } from "react";
import { entityColor } from "../../lib/colors";
import { KnowledgeEntityCard } from "./KnowledgeEntityCard";
import type { SearchResult } from "../../store/types";

interface EntityGroupSectionProps {
  type: string;
  entities: SearchResult[];
}

export function EntityGroupSection({ type, entities }: EntityGroupSectionProps) {
  const [collapsed, setCollapsed] = useState(false);
  const color = entityColor(type);

  return (
    <div style={{ marginBottom: 16 }}>
      {/* Group header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          padding: "6px 4px",
          background: "none",
          border: "none",
          cursor: "pointer",
          marginBottom: collapsed ? 0 : 8,
        }}
      >
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: 3,
            background: color,
            boxShadow: `0 0 8px ${color}30`,
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text-primary)", flex: 1, textAlign: "left" }}>
          {type}
        </span>
        <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-muted)" }}>
          {entities.length}
        </span>
        <span
          style={{
            fontSize: 10,
            color: "var(--text-muted)",
            transition: "transform 0.2s ease",
            transform: collapsed ? "rotate(-90deg)" : "rotate(0deg)",
          }}
        >
          {"\u25BE"}
        </span>
      </button>

      {/* Entity cards grid */}
      {!collapsed && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 8,
          }}
        >
          {entities.map((entity) => (
            <KnowledgeEntityCard key={entity.id} entity={entity} />
          ))}
        </div>
      )}
    </div>
  );
}
