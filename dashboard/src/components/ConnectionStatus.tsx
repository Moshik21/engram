import { useEngramStore } from "../store";
import type { WsReadyState } from "../store/types";

const STATE_CONFIG: Record<
  WsReadyState,
  { color: string; glow: string; label: string; pulse: boolean }
> = {
  connected: {
    color: "var(--accent)",
    glow: "var(--accent-glow-strong)",
    label: "Connected",
    pulse: false,
  },
  connecting: {
    color: "var(--warm)",
    glow: "rgba(251, 146, 60, 0.4)",
    label: "Connecting",
    pulse: true,
  },
  reconnecting: {
    color: "var(--warm)",
    glow: "rgba(251, 146, 60, 0.4)",
    label: "Reconnecting",
    pulse: true,
  },
  disconnected: {
    color: "#ef4444",
    glow: "rgba(239, 68, 68, 0.4)",
    label: "Offline",
    pulse: false,
  },
};

export function ConnectionStatus() {
  const readyState = useEngramStore((s) => s.readyState);
  const config = STATE_CONFIG[readyState];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        padding: "8px 14px",
      }}
    >
      <span
        className={config.pulse ? "animate-pulse-soft" : undefined}
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: config.color,
          boxShadow: `0 0 6px ${config.glow}`,
          transition: "all 0.3s ease",
        }}
      />
      <span
        className="mono"
        style={{
          fontSize: 9,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color:
            readyState === "connected"
              ? "var(--text-muted)"
              : config.color,
          transition: "color 0.3s ease",
        }}
      >
        {config.label}
      </span>
    </div>
  );
}
