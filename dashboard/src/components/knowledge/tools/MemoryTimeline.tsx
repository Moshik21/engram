import { formatRelativeTime } from "../../../lib/utils";

interface TimelineEpisode {
  id: string;
  content: string;
  source?: string | null;
  createdAt?: string | null;
  score: number;
}

export function MemoryTimeline({ episodes }: { episodes: TimelineEpisode[] }) {
  if (!episodes || episodes.length === 0) return null;

  return (
    <div
      style={{
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        padding: "10px 12px",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontFamily: "var(--font-mono)",
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          marginBottom: 10,
        }}
      >
        Memory Timeline
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {episodes.map((ep, i) => (
          <div
            key={ep.id}
            style={{
              display: "flex",
              gap: 10,
              paddingBottom: i < episodes.length - 1 ? 10 : 0,
              position: "relative",
            }}
          >
            {/* Timeline line + dot */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                flexShrink: 0,
                width: 12,
              }}
            >
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: "var(--accent)",
                  boxShadow: "0 0 6px var(--accent-glow)",
                  flexShrink: 0,
                  marginTop: 4,
                }}
              />
              {i < episodes.length - 1 && (
                <div
                  style={{
                    width: 1,
                    flex: 1,
                    background: "var(--border)",
                    marginTop: 4,
                  }}
                />
              )}
            </div>

            {/* Content */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                {ep.createdAt && (
                  <span
                    className="mono"
                    style={{ fontSize: 9, color: "var(--text-muted)" }}
                  >
                    {formatRelativeTime(ep.createdAt)}
                  </span>
                )}
                {ep.source && (
                  <span
                    style={{
                      fontSize: 9,
                      padding: "1px 5px",
                      borderRadius: "var(--radius-xs)",
                      background: "rgba(255,255,255,0.05)",
                      border: "1px solid var(--border)",
                      color: "var(--text-muted)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {ep.source}
                  </span>
                )}
                <span
                  className="mono"
                  style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: "auto" }}
                >
                  {(ep.score * 100).toFixed(0)}%
                </span>
              </div>
              <p
                style={{
                  margin: 0,
                  fontSize: 11,
                  color: "var(--text-secondary)",
                  lineHeight: 1.4,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  display: "-webkit-box",
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: "vertical",
                }}
              >
                {ep.content}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
