import type { RecallResult } from "../../store/types";
import { entityColor, entityColorDim } from "../../lib/colors";

export function RecallResultCard({ result }: { result: RecallResult }) {
  const score = result.score;
  const bd = result.scoreBreakdown;

  if (result.resultType === "entity") {
    const { entity, relationships } = result;
    const color = entityColor(entity.entityType);

    return (
      <div className="card card-glow" style={{ padding: "12px 14px", marginBottom: 8 }}>
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
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", flex: 1 }}>
            {entity.name}
          </span>
          <span
            className="mono tabular-nums"
            style={{
              fontSize: 10,
              padding: "2px 6px",
              borderRadius: 99,
              background: entityColorDim(entity.entityType, 0.08),
              color,
            }}
          >
            {entity.entityType}
          </span>
        </div>

        {entity.summary && (
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 6, lineHeight: 1.4 }}>
            {entity.summary}
          </p>
        )}

        {/* Score breakdown */}
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          {[
            { label: "SEM", value: bd.semantic },
            { label: "ACT", value: bd.activation },
            { label: "EDG", value: bd.edgeProximity },
          ].map(({ label, value }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span className="label" style={{ fontSize: 8 }}>{label}</span>
              <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                {value.toFixed(2)}
              </span>
            </div>
          ))}
          <div style={{ flex: 1 }} />
          <span className="mono tabular-nums" style={{ fontSize: 11, color: "var(--accent)" }}>
            {score.toFixed(3)}
          </span>
        </div>

        {/* Relationships */}
        {relationships.length > 0 && (
          <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
            {relationships.slice(0, 5).map((rel, i) => (
              <span
                key={i}
                className="mono"
                style={{
                  fontSize: 9,
                  padding: "2px 6px",
                  borderRadius: 99,
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid var(--border)",
                  color: "var(--text-muted)",
                }}
              >
                {rel.predicate}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Episode result
  const { episode } = result;
  return (
    <div className="card" style={{ padding: "12px 14px", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 2,
            background: "var(--text-muted)",
            flexShrink: 0,
          }}
        />
        <span className="mono" style={{ fontSize: 10, color: "var(--text-muted)" }}>
          Episode
        </span>
        <span style={{ flex: 1 }} />
        <span className="mono tabular-nums" style={{ fontSize: 9, color: "var(--text-ghost)" }}>
          {episode.source}
        </span>
      </div>

      <p style={{
        fontSize: 12,
        color: "var(--text-secondary)",
        marginTop: 6,
        lineHeight: 1.4,
        overflow: "hidden",
        textOverflow: "ellipsis",
        display: "-webkit-box",
        WebkitLineClamp: 4,
        WebkitBoxOrient: "vertical",
      }}>
        {episode.content}
      </p>

      {/* Score breakdown */}
      <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
        {[
          { label: "SEM", value: bd.semantic },
          { label: "ACT", value: bd.activation },
        ].map(({ label, value }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span className="label" style={{ fontSize: 8 }}>{label}</span>
            <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
              {value.toFixed(2)}
            </span>
          </div>
        ))}
        <div style={{ flex: 1 }} />
        <span className="mono tabular-nums" style={{ fontSize: 11, color: "var(--accent)" }}>
          {score.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
