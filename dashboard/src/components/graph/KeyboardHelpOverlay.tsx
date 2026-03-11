interface Props {
  onClose: () => void;
}

const SHORTCUTS = [
  { key: "/", action: "Open search" },
  { key: "Escape", action: "Deselect / close" },
  { key: "Space", action: "Center on selected node" },
  { key: "Alt + \u2190/\u2192", action: "History back / forward" },
  { key: "E", action: "Expand selected node" },
  { key: "R", action: "Reset camera (zoom to fit)" },
  { key: "H", action: "Toggle heatmap" },
  { key: "L", action: "Toggle edge labels" },
  { key: "G", action: "Toggle 2D / 3D" },
  { key: "?", action: "Show this help" },
] as const;

export function KeyboardHelpOverlay({ onClose }: Props) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 60,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(3, 4, 8, 0.7)",
        backdropFilter: "blur(4px)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "rgba(15, 17, 25, 0.95)",
          border: "1px solid rgba(99, 102, 241, 0.2)",
          borderRadius: 12,
          padding: "20px 28px",
          minWidth: 300,
          maxWidth: 400,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: "var(--accent, #a5b4fc)",
            marginBottom: 16,
            fontFamily: "var(--font-display)",
          }}
        >
          Keyboard Shortcuts
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <tbody>
            {SHORTCUTS.map((s) => (
              <tr key={s.key}>
                <td
                  style={{
                    padding: "5px 12px 5px 0",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "rgba(196, 181, 253, 0.9)",
                    whiteSpace: "nowrap",
                  }}
                >
                  <kbd
                    style={{
                      padding: "2px 6px",
                      borderRadius: 4,
                      background: "rgba(99, 102, 241, 0.15)",
                      border: "1px solid rgba(99, 102, 241, 0.25)",
                      fontSize: 11,
                    }}
                  >
                    {s.key}
                  </kbd>
                </td>
                <td
                  style={{
                    padding: "5px 0",
                    fontSize: 12,
                    color: "rgba(148, 163, 184, 0.8)",
                  }}
                >
                  {s.action}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div
          style={{
            marginTop: 16,
            fontSize: 11,
            color: "rgba(148, 163, 184, 0.4)",
            textAlign: "center",
          }}
        >
          Press Escape or click outside to close
        </div>
      </div>
    </div>
  );
}
