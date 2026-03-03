import type { Episode, EpisodeStatus } from "../store/types";
import { entityColor, entityColorDim } from "../lib/colors";
import { formatRelativeTime } from "../lib/utils";

const STATUS_STYLES: Record<
  EpisodeStatus,
  { bg: string; color: string; border: string; label: string }
> = {
  queued: {
    bg: "rgba(255,255,255,0.04)",
    color: "var(--text-secondary)",
    border: "var(--border)",
    label: "Queued",
  },
  processing: {
    bg: "rgba(251, 146, 60, 0.08)",
    color: "var(--warm)",
    border: "rgba(251, 146, 60, 0.25)",
    label: "Processing",
  },
  completed: {
    bg: "rgba(34, 211, 238, 0.06)",
    color: "var(--accent)",
    border: "var(--border-active)",
    label: "Completed",
  },
  failed: {
    bg: "rgba(239, 68, 68, 0.08)",
    color: "#ef4444",
    border: "rgba(239, 68, 68, 0.25)",
    label: "Failed",
  },
};

interface EpisodeCardProps {
  episode: Episode;
  onEntityClick: (entityId: string) => void;
}

export function EpisodeCard({ episode, onEntityClick }: EpisodeCardProps) {
  const statusStyle = STATUS_STYLES[episode.status];
  const preview =
    episode.content.length > 120
      ? episode.content.slice(0, 120) + "..."
      : episode.content;

  return (
    <div
      className="card animate-fade-in"
      style={{
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span
          className="pill"
          style={{
            borderColor: statusStyle.border,
            background: statusStyle.bg,
            color: statusStyle.color,
          }}
        >
          <span
            style={{
              width: 4,
              height: 4,
              borderRadius: "50%",
              background: statusStyle.color,
            }}
          />
          {statusStyle.label}
        </span>
        <span className="pill">
          {episode.source}
        </span>
        <span style={{ flex: 1 }} />
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--text-muted)" }}
        >
          {formatRelativeTime(episode.createdAt)}
        </span>
      </div>

      {/* Content preview */}
      <p
        style={{
          margin: 0,
          fontSize: 12,
          lineHeight: 1.6,
          color: "var(--text-secondary)",
        }}
      >
        {preview}
      </p>

      {/* Error message */}
      {episode.error && (
        <p
          className="mono"
          style={{
            margin: 0,
            fontSize: 10,
            color: "#ef4444",
            background: "rgba(239, 68, 68, 0.06)",
            padding: "4px 8px",
            borderRadius: "var(--radius-xs)",
            border: "1px solid rgba(239, 68, 68, 0.15)",
          }}
        >
          {episode.error}
        </p>
      )}

      {/* Entity chips */}
      {episode.entities.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {episode.entities.map((ent) => {
            const color = entityColor(ent.entityType);
            return (
              <button
                key={ent.id}
                onClick={() => onEntityClick(ent.id)}
                className="pill"
                style={{
                  cursor: "pointer",
                  borderColor: `${color}25`,
                  background: entityColorDim(ent.entityType, 0.08),
                  color,
                  transition: "all 0.15s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = entityColorDim(
                    ent.entityType,
                    0.15,
                  );
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = entityColorDim(
                    ent.entityType,
                    0.08,
                  );
                }}
              >
                <span
                  style={{
                    width: 4,
                    height: 4,
                    borderRadius: "50%",
                    background: color,
                  }}
                />
                {ent.name}
              </button>
            );
          })}
        </div>
      )}

      {/* Facts count */}
      {episode.factsCount > 0 && (
        <div
          className="mono"
          style={{ fontSize: 10, color: "var(--text-muted)" }}
        >
          {episode.factsCount} fact{episode.factsCount !== 1 ? "s" : ""}{" "}
          extracted
        </div>
      )}
    </div>
  );
}
