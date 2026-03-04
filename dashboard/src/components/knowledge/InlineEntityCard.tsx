import { useEngramStore } from "../../store";
import { entityColor, entityColorDim } from "../../lib/colors";

export interface InlineEntitySource {
  type: "entity" | "episode";
  id: string;
  name: string;
  entityType?: string;
  summary?: string | null;
  content?: string;
  score?: number;
}

export function InlineEntityCard({ source }: { source: InlineEntitySource }) {
  const openDrawer = useEngramStore((s) => s.openDrawer);

  if (source.type === "episode") {
    // Episode chip — non-clickable
    return (
      <div
        style={{
          padding: "6px 10px",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border)",
          background: "var(--surface)",
          fontSize: 11,
          color: "var(--text-secondary)",
          maxWidth: 260,
          lineHeight: 1.4,
        }}
      >
        <div
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
          }}
        >
          {source.content || source.name}
        </div>
        {source.score != null && (
          <span
            className="mono"
            style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2, display: "block" }}
          >
            {(source.score * 100).toFixed(0)}% match
          </span>
        )}
      </div>
    );
  }

  // Entity card — clickable
  const color = entityColor(source.entityType ?? "Other");

  return (
    <button
      onClick={() => {
        if (source.id) openDrawer(source.id);
      }}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 3,
        padding: "6px 10px",
        borderRadius: "var(--radius-sm)",
        border: `1px solid ${color}18`,
        background: entityColorDim(source.entityType ?? "Other", 0.06),
        cursor: source.id ? "pointer" : "default",
        textAlign: "left",
        maxWidth: 200,
        transition: "all 0.15s ease",
        fontFamily: "var(--font-body)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = `${color}35`;
        e.currentTarget.style.background = entityColorDim(source.entityType ?? "Other", 0.12);
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = `${color}18`;
        e.currentTarget.style.background = entityColorDim(source.entityType ?? "Other", 0.06);
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 11,
            fontWeight: 500,
            color: "var(--text-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {source.name}
        </span>
      </div>
      {source.summary && (
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
          {source.summary}
        </span>
      )}
      {source.score != null && (
        <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)" }}>
          {(source.score * 100).toFixed(0)}%
        </span>
      )}
    </button>
  );
}
