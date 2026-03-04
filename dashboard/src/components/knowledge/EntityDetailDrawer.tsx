import { useState } from "react";
import { useEngramStore } from "../../store";
import { entityColor, entityColorDim, activationColor } from "../../lib/colors";
import { formatRelativeTime } from "../../lib/utils";

export function EntityDetailDrawer() {
  const drawerEntity = useEngramStore((s) => s.drawerEntity);
  const isDrawerLoading = useEngramStore((s) => s.isDrawerLoading);
  const closeDrawer = useEngramStore((s) => s.closeDrawer);
  const openDrawer = useEngramStore((s) => s.openDrawer);
  const updateEntity = useEngramStore((s) => s.updateEntity);
  const selectNode = useEngramStore((s) => s.selectNode);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);
  const setConfirmDialog = useEngramStore((s) => s.setConfirmDialog);

  const [editing, setEditing] = useState(false);
  const [editSummary, setEditSummary] = useState("");

  const detail = drawerEntity;
  const typeColor = detail ? entityColor(detail.entityType) : "#64748b";

  return (
    <div
      className="card animate-slide-right"
      style={{
        position: "absolute",
        right: 0,
        top: 0,
        bottom: 0,
        zIndex: 30,
        width: 320,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        borderRadius: 0,
        borderLeft: "1px solid var(--border)",
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
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: typeColor }} />
              {detail.entityType}
            </span>
          )}
        </div>
        <button
          onClick={closeDrawer}
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
          onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-primary)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
        >
          &times;
        </button>
      </div>

      {isDrawerLoading && (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="skeleton" style={{ width: 100, height: 14 }} />
        </div>
      )}

      {detail && !isDrawerLoading && (
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
          {/* Activation gauge */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
              <span className="label">Activation</span>
              <span
                className="mono tabular-nums"
                style={{ fontSize: 11, fontWeight: 500, color: activationColor(detail.activationCurrent) }}
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

          {/* Editable summary */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span className="label">Summary</span>
              {!editing && (
                <button
                  onClick={() => {
                    setEditSummary(detail.summary || "");
                    setEditing(true);
                  }}
                  className="mono"
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                    fontSize: 9,
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = "var(--accent)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
                >
                  Edit
                </button>
              )}
            </div>
            {editing ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <textarea
                  value={editSummary}
                  onChange={(e) => setEditSummary(e.target.value)}
                  style={{
                    width: "100%",
                    minHeight: 60,
                    padding: "8px 10px",
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    color: "var(--text-primary)",
                    fontSize: 12,
                    fontFamily: "var(--font-body)",
                    lineHeight: 1.5,
                    resize: "vertical",
                    outline: "none",
                  }}
                />
                <div style={{ display: "flex", gap: 6 }}>
                  <button
                    onClick={async () => {
                      await updateEntity(detail.id, { summary: editSummary });
                      setEditing(false);
                    }}
                    style={{
                      flex: 1,
                      padding: "5px 0",
                      background: "rgba(34, 211, 238, 0.08)",
                      border: "1px solid var(--border-active)",
                      borderRadius: "var(--radius-xs)",
                      color: "var(--accent)",
                      fontSize: 10,
                      fontFamily: "var(--font-mono)",
                      cursor: "pointer",
                    }}
                  >
                    Save
                  </button>
                  <button
                    onClick={() => setEditing(false)}
                    style={{
                      flex: 1,
                      padding: "5px 0",
                      background: "transparent",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-xs)",
                      color: "var(--text-muted)",
                      fontSize: 10,
                      fontFamily: "var(--font-mono)",
                      cursor: "pointer",
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
                {detail.summary || "No summary"}
              </p>
            )}
          </div>

          {/* Stats row */}
          <div style={{ display: "flex", gap: 8 }}>
            <div style={{ flex: 1, padding: "7px 10px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}>
              <div className="label" style={{ marginBottom: 2 }}>Accesses</div>
              <div className="mono tabular-nums" style={{ fontSize: 15, color: "#fff" }}>
                {detail.accessCount}
              </div>
            </div>
            <div style={{ flex: 1, padding: "7px 10px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}>
              <div className="label" style={{ marginBottom: 2 }}>Last Seen</div>
              <div className="mono" style={{ fontSize: 12, color: "#fff" }}>
                {formatRelativeTime(detail.lastAccessed)}
              </div>
            </div>
          </div>

          {/* Relationships */}
          {detail.facts.length > 0 && (
            <div>
              <div className="label" style={{ marginBottom: 8 }}>
                Relationships ({detail.facts.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
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
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--border-hover)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-subtle)"; }}
                  >
                    <span style={{ color: entityColor(fact.other.entityType), fontSize: 9 }}>
                      {fact.direction === "outgoing" ? "\u2192" : "\u2190"}
                    </span>
                    <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)", flexShrink: 0 }}>
                      {fact.predicate}
                    </span>
                    <button
                      onClick={() => openDrawer(fact.other.id)}
                      style={{
                        background: "none",
                        border: "none",
                        color: entityColor(fact.other.entityType),
                        fontSize: 11,
                        cursor: "pointer",
                        padding: 0,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        fontFamily: "var(--font-body)",
                      }}
                    >
                      {fact.other.name}
                    </button>
                    {fact.validFrom && (
                      <span className="mono" style={{ fontSize: 8, color: "var(--text-ghost)", marginLeft: "auto" }}>
                        {new Date(fact.validFrom).getFullYear()}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: "flex", gap: 6, marginTop: "auto" }}>
            <button
              onClick={() => {
                selectNode(detail.id);
                setCurrentView("graph");
                closeDrawer();
              }}
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
            >
              View in Graph
            </button>
            <button
              onClick={() => {
                setConfirmDialog({
                  type: "delete",
                  entityId: detail.id,
                  entityName: detail.name,
                  title: "Delete Entity",
                  message: `Are you sure you want to delete "${detail.name}"? This action cannot be undone.`,
                });
              }}
              style={{
                padding: "6px 12px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid rgba(248, 113, 113, 0.2)",
                background: "rgba(248, 113, 113, 0.06)",
                color: "#f87171",
                fontFamily: "var(--font-body)",
                fontSize: 12,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
