import { useCallback, useEffect, useState } from "react";
import { useEngramStore } from "../store";
import { api } from "../api/client";
import { entityColor, entityColorDim, activationColor } from "../lib/colors";
import { formatRelativeTime } from "../lib/utils";
import type { EntityDetail } from "../store/types";

export function NodeDetailPanel() {
  const selectedNodeId = useEngramStore((s) => s.selectedNodeId);
  const selectNode = useEngramStore((s) => s.selectNode);
  const loadNeighborhood = useEngramStore((s) => s.loadNeighborhood);
  const expandNode = useEngramStore((s) => s.expandNode);

  const [detail, setDetail] = useState<EntityDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedNodeId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .getEntity(selectedNodeId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNodeId]);

  const handleEntityClick = useCallback(
    (entityId: string) => {
      loadNeighborhood(entityId);
      selectNode(entityId);
    },
    [loadNeighborhood, selectNode]
  );

  if (!selectedNodeId) return null;

  const typeColor = detail ? entityColor(detail.entityType) : "#64748b";

  return (
    <div
      className="card animate-slide-right"
      style={{
        position: "absolute",
        right: 10,
        top: 54,
        bottom: 10,
        zIndex: 20,
        width: 280,
        borderRadius: "var(--radius-lg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "14px 14px 10px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          position: "relative",
        }}
      >
        {/* Subtle type-colored top line */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 14,
            right: 14,
            height: 1,
            background: `linear-gradient(90deg, transparent, ${typeColor}40, transparent)`,
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <h2
            className="display"
            style={{
              fontSize: 20,
              fontWeight: 400,
              color: "#fff",
              margin: 0,
              lineHeight: 1.2,
            }}
          >
            {detail?.name ?? "..."}
          </h2>
          {detail && (
            <span
              className="pill"
              style={{
                marginTop: 6,
                display: "inline-flex",
                borderColor: `${typeColor}25`,
                background: entityColorDim(detail.entityType, 0.08),
                color: typeColor,
              }}
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: typeColor,
                }}
              />
              {detail.entityType}
            </span>
          )}
        </div>
        <button
          onClick={() => selectNode(null)}
          style={{
            background: "none",
            border: "none",
            color: "var(--text-muted)",
            cursor: "pointer",
            fontSize: 16,
            lineHeight: 1,
            padding: "2px 4px",
            transition: "color 0.15s",
            borderRadius: "var(--radius-xs)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = "var(--text-muted)";
          }}
        >
          &times;
        </button>
      </div>

      {loading && (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div className="skeleton" style={{ width: 100, height: 14 }} />
        </div>
      )}

      {detail && !loading && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: 14,
            display: "flex",
            flexDirection: "column",
            gap: 14,
          }}
        >
          {/* Summary */}
          {detail.summary && (
            <p
              style={{
                fontSize: 12,
                color: "var(--text-secondary)",
                lineHeight: 1.6,
                margin: 0,
              }}
            >
              {detail.summary}
            </p>
          )}

          {/* Activation bar */}
          <div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 5,
              }}
            >
              <span className="label">Activation</span>
              <span
                className="mono tabular-nums"
                style={{
                  fontSize: 11,
                  fontWeight: 500,
                  color: activationColor(detail.activationCurrent),
                }}
              >
                {(detail.activationCurrent * 100).toFixed(1)}%
              </span>
            </div>
            <div className="metric-bar">
              <div
                className="metric-bar-fill"
                style={{
                  width: `${detail.activationCurrent * 100}%`,
                  background: `linear-gradient(90deg, var(--accent), ${activationColor(detail.activationCurrent)})`,
                }}
              />
            </div>
          </div>

          {/* Stats row */}
          <div style={{ display: "flex", gap: 8 }}>
            <div
              style={{
                flex: 1,
                padding: "7px 10px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
              }}
            >
              <div className="label" style={{ marginBottom: 2 }}>
                Accesses
              </div>
              <div
                className="mono tabular-nums"
                style={{ fontSize: 15, color: "#fff" }}
              >
                {detail.accessCount}
              </div>
            </div>
            <div
              style={{
                flex: 1,
                padding: "7px 10px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
              }}
            >
              <div className="label" style={{ marginBottom: 2 }}>
                Last Seen
              </div>
              <div
                className="mono"
                style={{ fontSize: 12, color: "#fff" }}
              >
                {formatRelativeTime(detail.lastAccessed)}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => expandNode(detail.id)}
              style={{
                flex: 1,
                padding: "6px 0",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--text-secondary)",
                fontFamily: "var(--font-body)",
                fontSize: 12,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "var(--border-hover)";
                e.currentTarget.style.color = "var(--text-primary)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border)";
                e.currentTarget.style.color = "var(--text-secondary)";
              }}
            >
              Expand
            </button>
            <button
              onClick={() => loadNeighborhood(detail.id)}
              style={{
                flex: 1,
                padding: "6px 0",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border-active)",
                background: "rgba(34, 211, 238, 0.06)",
                color: "var(--accent)",
                fontFamily: "var(--font-body)",
                fontSize: 12,
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background =
                  "rgba(34, 211, 238, 0.12)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background =
                  "rgba(34, 211, 238, 0.06)";
              }}
            >
              Center
            </button>
          </div>

          {/* Relationships */}
          {detail.facts.length > 0 && (
            <div>
              <div className="label" style={{ marginBottom: 8 }}>
                Relationships ({detail.facts.length})
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                }}
              >
                {detail.facts.map((fact) => (
                  <div
                    key={fact.id}
                    style={{
                      padding: "6px 8px",
                      borderRadius: "var(--radius-xs)",
                      border: "1px solid var(--border-subtle)",
                      transition: "border-color 0.15s",
                      display: "flex",
                      alignItems: "center",
                      gap: 5,
                      fontSize: 11,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor =
                        "var(--border-hover)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor =
                        "var(--border-subtle)";
                    }}
                  >
                    {fact.direction === "outgoing" ? (
                      <>
                        <span
                          className="mono"
                          style={{
                            fontSize: 9,
                            color: "var(--text-muted)",
                            flexShrink: 0,
                          }}
                        >
                          {fact.predicate}
                        </span>
                        <span
                          style={{
                            color: "var(--text-ghost)",
                            fontSize: 9,
                          }}
                        >
                          &rarr;
                        </span>
                        <button
                          onClick={() => handleEntityClick(fact.other.id)}
                          style={{
                            background: "none",
                            border: "none",
                            color: "var(--accent)",
                            fontSize: 11,
                            cursor: "pointer",
                            padding: 0,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {fact.other.name}
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => handleEntityClick(fact.other.id)}
                          style={{
                            background: "none",
                            border: "none",
                            color: "var(--accent)",
                            fontSize: 11,
                            cursor: "pointer",
                            padding: 0,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {fact.other.name}
                        </button>
                        <span
                          style={{
                            color: "var(--text-ghost)",
                            fontSize: 9,
                          }}
                        >
                          &rarr;
                        </span>
                        <span
                          className="mono"
                          style={{
                            fontSize: 9,
                            color: "var(--text-muted)",
                            flexShrink: 0,
                          }}
                        >
                          {fact.predicate}
                        </span>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
