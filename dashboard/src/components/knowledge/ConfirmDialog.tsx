import { useEngramStore } from "../../store";

export function ConfirmDialog() {
  const confirmDialog = useEngramStore((s) => s.confirmDialog);
  const setConfirmDialog = useEngramStore((s) => s.setConfirmDialog);
  const confirmAction = useEngramStore((s) => s.confirmAction);

  if (!confirmDialog) return null;

  return (
    <div
      onClick={() => setConfirmDialog(null)}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 60,
        background: "rgba(3, 4, 8, 0.7)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        className="glass-elevated animate-slide-up"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 340,
          borderRadius: "var(--radius-lg)",
          padding: 20,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 500, color: "var(--text-primary)", margin: 0 }}>
          {confirmDialog.title}
        </h3>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
          {confirmDialog.message}
        </p>
        <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
          <button
            onClick={() => setConfirmDialog(null)}
            style={{
              flex: 1,
              padding: "8px 0",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-secondary)",
              fontFamily: "var(--font-body)",
              fontSize: 12,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            Cancel
          </button>
          <button
            onClick={confirmAction}
            style={{
              flex: 1,
              padding: "8px 0",
              borderRadius: "var(--radius-sm)",
              border: "1px solid rgba(248, 113, 113, 0.3)",
              background: "rgba(248, 113, 113, 0.1)",
              color: "#f87171",
              fontFamily: "var(--font-body)",
              fontSize: 12,
              fontWeight: 500,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}
